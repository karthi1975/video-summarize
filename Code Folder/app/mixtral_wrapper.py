import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from .mixtral_prompts import get_quiz_prompt, get_interview_prompt, get_correction_prompt
from .utils import chunk_transcription, parse_json_output
from huggingface_hub import login

class MixtralWrapper:
    def __init__(self,
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        use_cuda=True,
        model_dir='mixtral', 
        max_chunk_size=27000
    ):
        """
        Initializes the model pipeline for further usage.
        """
        
        login(token="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")
        self.model_id = model_dir if os.path.exists(model_dir) else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.max_chunk_size = max_chunk_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_dir = model_dir

        # Dynamically select a device to run on depending on whether CUDA is available or not.
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype=torch.float16)  # Use FP16 to reduce memory usage
        else:
            self.device = torch.device("cpu")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)

    def tokenize(self, examples):
        """
        Tokenizes raw text inputs into tokens using the pretrained tokenizer.
        """
        # Adjust tokenization to use 'input_text' for the model inputs
        model_inputs = self.tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=30000)

        # Setup the tokenizer for targets using 'target_text'
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=30000)

        model_inputs["labels"] = labels["input_ids"]

        # Move model_inputs to the appropriate device (GPU or CPU)
        model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

        return model_inputs

    def fine_tune_model(self, data_path):
        """
        Fine tune the model by training it on a given dataset.
        """
        # Loading datasets
        datasets = load_dataset('csv', data_files={'train': f'{data_path}/train.csv', 'validation': f'{data_path}/validation.csv'})
        
        tokenized_datasets = datasets.map(self.tokenize, batched=True)
        
        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=1,              
            per_device_train_batch_size=2,  
            per_device_eval_batch_size=2,   
            warmup_steps=500,                
            weight_decay=0.01,               
            logging_dir=os.path.join(self.model_dir, 'logs'),            
            logging_steps=10,
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,                         
            args=training_args,                  
            train_dataset=tokenized_datasets['train'],         
            eval_dataset=tokenized_datasets['validation'],
        )
        
        # Train the model
        trainer.train()
        
        # Save the fine-tuned model and tokenizer
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    # def generate(self, messages, max_new_tokens=5000):
    #     """
    #     Generates a model response based on the given messages.
    #     """
    #     model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True, truncation=True).to(self.device)

    #     # Add attention mask and set pad_token_id to avoid warning
    #     model_inputs['attention_mask'] = model_inputs['input_ids'].ne(self.tokenizer.pad_token_id)
    #     self.model.config.pad_token_id = self.tokenizer.eos_token_id

    #     generated_ids = self.model.generate(model_inputs["input_ids"], attention_mask=model_inputs['attention_mask'], max_new_tokens=max_new_tokens, do_sample=True)
    #     return self.tokenizer.batch_decode(generated_ids)[0].split('[/INST]')[1]  # Parse the output format of the Mixtral model.

    def generate(self, messages, max_new_tokens=5000):
        """
        Generates a model response based on the given messages.
        """
        # Ensure `messages` is a string or list of strings
        if isinstance(messages, str):
            pass  # `messages` is already a valid input
        elif isinstance(messages, list):
            # Ensure that all elements of the list are strings
            messages = [str(m) for m in messages]
        else:
            raise ValueError(f"Invalid message format. Expected str or List[str], but got {type(messages)}")

        # Tokenize the messages
        model_inputs = self.tokenizer(
            messages,  # Ensure it's a valid string or list of strings
            return_tensors="pt",  # Return PyTorch tensors
            padding=True,  # Ensure padding
            truncation=True,  # Ensure truncation
            max_length=1024,  # Set a max length to avoid truncation issues
            return_attention_mask=True  # Generate the attention mask
        ).to(self.device)

        # Set pad_token_id to eos_token_id to avoid warnings
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Generate the output
        generated_ids = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True
        )

        # Decode the output and return the text
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
      


    def correct_transcription(self, transcription):
        """
        Runs the given transcription through the model to fix any errors in spelling or grammar.
        """
        queue = chunk_transcription(self.tokenizer, transcription, self.max_chunk_size)
        result = []
        for chunk in queue:
            messages = get_correction_prompt(' '.join(chunk))
            result.append(self.generate(messages))

        return ' '.join(result)

    def generate_quiz(self, transcription, complexity='medium', q_type='quiz'):
        """
        Generates a quiz based on video transcription, complexity and quiz type.
        Returns a list with questions and answers.
        """
        messages = get_quiz_prompt(complexity, transcription) if q_type == 'quiz' else get_interview_prompt(complexity, transcription)
        output = self.generate(messages)
        quiz_json = parse_json_output(output)

        return quiz_json