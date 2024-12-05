from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

def preprocess_data(raw_data):
    data = []
    for article in raw_data["train"]["data"]:  # Iterate directly over the 'data' list
        for item in article:
            paragraphs = item.get("paragraphs", [])  # Paragraph is a list within each article
            for paragraph in paragraphs:
                context = paragraph.get("context", "")
                for qa in paragraph.get("qas", []):  # Iterate over the 'qas' list
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    if answers:  # Only access the first answer if the list is not empty
                        answer = answers[0].get("text", "")  # Get the text of the first answer
                    else:
                        answer = ""  # If no answer is found, set to an empty string
                    data.append({
                        "context": context,
                        "question": question,
                        "answer": answer,
                    })
    return data

def tokenize_function(examples, tokenizer):
    inputs = [
        f"Context: {context} Question: {question} Answer: {answer}"
        for context, question, answer in zip(examples["context"], examples["question"], examples["answer"])
    ]
    return tokenizer(inputs, truncation=True, padding="max_length", max_length=512)

def main():
    # Load the dataset
    dataset = load_dataset('json', data_files='train-v2.json')
    
    # Preprocess the dataset
    processed_data = preprocess_data(dataset)
    dataset = Dataset.from_list(processed_data)
    
    # Initialize tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    
    # Tokenize the data
    subset_data = dataset.select(range(20000))
    tokenized_datasets = subset_data.map(
        lambda examples: tokenize_function(examples, gpt_tokenizer), batched=True
    )
    
    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=gpt_tokenizer,
        mlm=False
    )
    
    # Load pre-trained GPT2 model
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",             # where to save the model checkpoints
        overwrite_output_dir=True,
        num_train_epochs=2,                 # number of epochs
        per_device_train_batch_size=4,      # batch size per GPU/TPU core
        save_steps=10_000,                  # save model after these many steps
        save_total_limit=2,                 # limit to save only 2 checkpoints
        logging_dir="./logs",               # directory for logs
        report_to="none"
    )
    
    # Define the Trainer
    trainer = Trainer(
        model=gpt_model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=gpt_tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Save the fine-tuned model
    gpt_model.save_pretrained("./fine_tuned_gpt2_qa")
    gpt_tokenizer.save_pretrained("./fine_tuned_gpt2_qa")

if __name__ == "__main__":
    main()