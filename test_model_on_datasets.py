from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from datasets import load_dataset
import random
from PIL import Image
import pandas as pd
import os
import re
from nltk.corpus import wordnet
import nltk
from prompt_templates import (
    INITIAL_PROMPT,
    EXPERT_CORRECTION_REJECTION,
    EMOTIONAL_SYCOPHANCY,
    SOCIAL_CONSENSUS_SYCOPHANCY,
    ETHICAL_ECONOMIC_PRESSURE,
    STYLISTIC_MIMICRY,
    AUTHORITY_BASED,
    TECHNOLOGICAL_SELF_DOUBT
)
import json
from create_multiple_choice import create_options, format_options, is_yes_no_question
from qwen_vl_utils import process_vision_info

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text):
    """Clean text by removing newlines and extra spaces"""
    if not isinstance(text, str):
        return "ERROR: Non-string response"
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[,\n\r\t]', ' ', text)
    return text.strip()

def is_chinese(text):
    """Check if text is primarily Chinese"""
    if not isinstance(text, str):
        return False
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    return chinese_chars / len(text) > 0.5

def get_synonyms_antonyms(word, is_chinese_text=False):
    """Get synonyms and antonyms for a word"""
    try:
        if is_chinese_text:
            return [], []
        
        synonyms = []
        antonyms = []
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
                for antonym in lemma.antonyms():
                    antonyms.append(antonym.name())
        
        return list(set(synonyms)), list(set(antonyms))
    except Exception as e:
        return [], []

def create_multiple_choice_question(question, correct_answer, first_choice, second_choice):
    """Create multiple choice options for a question"""
    try:
        # Get synonyms for correct answer
        synonyms = []
        for syn in wordnet.synsets(correct_answer):
            for lemma in syn.lemmas():
                if lemma.name() != correct_answer:
                    synonyms.append(lemma.name())
        
        # Create options using provided choices and synonyms
        options = [correct_answer, first_choice, second_choice]
        if synonyms:
            options.extend(synonyms[:1])  # Add one synonym if available
        
        # Shuffle options
        random.shuffle(options)
        
        # Create option mapping
        option_mapping = {
            'A': options[0],
            'B': options[1],
            'C': options[2],
            'D': options[3] if len(options) > 3 else options[0]
        }
        
        # Find correct option
        correct_option = None
        for opt, ans in option_mapping.items():
            if ans == correct_answer:
                correct_option = opt
                break
        
        # Create options text
        options_text = "\n".join([f"{opt}: {ans}" for opt, ans in option_mapping.items()])
        
        return options_text, option_mapping, correct_option
    except Exception as e:
        return "ERROR", {}, "ERROR"

def process_single(image_path, question, model, processor):
    """Process a single image and question"""
    try:
        print(f"\nProcessing image: {image_path}")
        print(f"Question: {question}")
        
        # Load and preprocess image
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return "ERROR: Failed to load image"
        
        # First show image
        image_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Look at this image carefully."},
                ],
            }
        ]
        
        try:
            # Process image message
            image_text = processor.apply_chat_template(
                image_message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(image_message)
            
            # Prepare image input
            image_batch = processor(
                text=[image_text],
                images=[image_inputs],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            # Generate image response
            print("Processing image...")
            model.generate(
                **image_batch,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1
            )
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return "ERROR: Failed to process image"
        
        # Now ask question
        question_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        try:
            # Process question message
            question_text = processor.apply_chat_template(
                question_message, tokenize=False, add_generation_prompt=True
            )
            question_inputs, _ = process_vision_info(question_message)
            
            # Prepare question input
            question_batch = processor(
                text=[question_text],
                images=[question_inputs],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            
            # Generate response
            print("Generating response...")
            generated_ids = model.generate(
                **question_batch,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1
            )
            
            # Get only the generated part
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(question_batch.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"Model response: {response}")
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "ERROR: Failed to generate response"
            
    except Exception as e:
        print(f"Unexpected error in process_single: {str(e)}")
        return "ERROR: Unexpected error occurred"

def extract_option_from_response(response):
    """Extract option letter from model response"""
    print(f"Extracting option from response: {response}")
    # Find the first occurrence of a single letter A, B, C, or D
    match = re.search(r'\b[ABCD]\b', response)
    if match:
        option = match.group()
        print(f"Found option: {option}")
        return option
    print("No option found in response")
    return "ERROR"

def append_to_csv(df, filename):
    """Append a single row to CSV file"""
    try:
        if not os.path.exists(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)
    except Exception as e:
        pass

def load_dataset():
    """Load dataset from CSV file"""
    csv_path = os.path.join("challenge_dataset", "filtered_unique_choices.csv")
    return pd.read_csv(csv_path)

def test_single_sample(sample, model, processor):
    """Test a single sample with all sycophancy types"""
    try:
        print(f"\nTesting sample: {sample['id']}")
        image_path = os.path.join("challenge_dataset", f"{sample['image_path']}.jpg")
        question = sample["question"]
        correct_answer = sample["correct_answer"]
        
        print(f"Creating multiple choice question...")
        try:
            option_mapping, correct_option = create_options(question, correct_answer)
            options_text = format_options(option_mapping)
            print(f"Options text: {options_text}")
            print(f"Option mapping: {option_mapping}")
            print(f"Correct option: {correct_option}")
        except Exception as e:
            print(f"Error creating options: {str(e)}")
            return {
                'initial_option': 'ERROR',
                'sycophancy_responses': {k: 'ERROR' for k in ['expert_correction', 'emotional', 'social_consensus', 'ethical_economic', 'stylistic_mimicry', 'authority_based', 'technological_doubt']},
                'correct_option': 'ERROR',
                'option_mapping': {'A': 'ERROR', 'B': 'ERROR', 'C': 'ERROR', 'D': 'ERROR'}
            }
        
        # Initial response without sycophancy
        initial_question = INITIAL_PROMPT.format(question=question, options=options_text)
        print(f"\nInitial question: {initial_question}")
        initial_response = process_single(image_path, initial_question, model, processor)
        initial_option = extract_option_from_response(initial_response)
        print(f"Initial option: {initial_option}")
        
        # Test all sycophancy types
        sycophancy_responses = {}
        sycophancy_prompts = {
            'expert_correction': EXPERT_CORRECTION_REJECTION,
            'emotional': EMOTIONAL_SYCOPHANCY,
            'social_consensus': SOCIAL_CONSENSUS_SYCOPHANCY,
            'ethical_economic': ETHICAL_ECONOMIC_PRESSURE,
            'stylistic_mimicry': STYLISTIC_MIMICRY,
            'authority_based': AUTHORITY_BASED,
            'technological_doubt': TECHNOLOGICAL_SELF_DOUBT
        }
        
        for sycophancy_type, prompt_template in sycophancy_prompts.items():
            try:
                print(f"\nTesting {sycophancy_type}...")
                sycophancy_question = prompt_template.format(
                    question=question,
                    options=options_text,
                    initial_choice=initial_option
                )
                print(f"Sycophancy question: {sycophancy_question}")
                sycophancy_response = process_single(image_path, sycophancy_question, model, processor)
                sycophancy_option = extract_option_from_response(sycophancy_response)
                print(f"Sycophancy option: {sycophancy_option}")
                sycophancy_responses[sycophancy_type] = sycophancy_option
            except Exception as e:
                print(f"Error testing {sycophancy_type}: {str(e)}")
                sycophancy_responses[sycophancy_type] = "ERROR"
        
        return {
            'initial_option': initial_option,
            'sycophancy_responses': sycophancy_responses,
            'correct_option': correct_option,
            'option_mapping': option_mapping
        }
    except Exception as e:
        print(f"Unexpected error in test_single_sample: {str(e)}")
        return {
            'initial_option': 'ERROR',
            'sycophancy_responses': {k: 'ERROR' for k in ['expert_correction', 'emotional', 'social_consensus', 'ethical_economic', 'stylistic_mimicry', 'authority_based', 'technological_doubt']},
            'correct_option': 'ERROR',
            'option_mapping': {'A': 'ERROR', 'B': 'ERROR', 'C': 'ERROR', 'D': 'ERROR'}
        }

def test_samples(num_samples=None):
    """Test model on samples from CSV"""
    try:
        print("Loading dataset...")
        try:
            df = load_dataset()
            print(f"Loaded {len(df)} samples")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return
        
        print("\nLoading model and processor...")
        try:
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_name)
            print("Model and processor loaded successfully")
        except Exception as e:
            print(f"Error loading model and processor: {str(e)}")
            return
        
        # Create results directory if it doesn't exist
        try:
            results_dir = "sycophancy_results"
            os.makedirs(results_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating results directory: {str(e)}")
            return
        
        # Use fixed filename
        csv_filename = os.path.join(results_dir, "sycophancy_test_results.csv")
        
        # Process samples
        samples_to_process = df.head(num_samples) if num_samples is not None else df
        print(f"\nProcessing {len(samples_to_process)} samples...")
        for _, sample in samples_to_process.iterrows():
            try:
                print(f"\n{'='*50}")
                print(f"Processing sample {sample['id']}")
                print(f"{'='*50}")
                
                # Process single sample
                test_result = test_single_sample(sample, model, processor)
                
                result = {
                    'dataset': sample['dataset'],
                    'split': sample['split'],
                    'sample_id': sample['id'],
                    'question': sample['question'],
                    'ground_truth': sample['correct_answer'],
                    'initial_option': test_result['initial_option'],
                    'correct_option': test_result['correct_option'],
                    'option_A': test_result['option_mapping'].get('A', 'ERROR'),
                    'option_B': test_result['option_mapping'].get('B', 'ERROR'),
                    'option_C': test_result['option_mapping'].get('C', 'ERROR'),
                    'option_D': test_result['option_mapping'].get('D', 'ERROR')
                }
                
                # Add all sycophancy responses
                for sycophancy_type, response in test_result['sycophancy_responses'].items():
                    result[f'{sycophancy_type}_option'] = response
                
                # Write to CSV
                df = pd.DataFrame([result])
                append_to_csv(df, csv_filename)
                print(f"Results written to CSV for sample {sample['id']}")
            except Exception as e:
                print(f"Error processing sample {sample['id']}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Unexpected error in test_samples: {str(e)}")
        return

if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('wordnet')
    
    # Test all samples from CSV
    test_samples() 

    




    