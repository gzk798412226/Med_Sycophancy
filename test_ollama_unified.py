#!/usr/bin/env python3
"""
Unified Ollama Vision-Language Model Sycophancy Testing Script

Usage:
    python test_ollama_unified.py --model llava:7b --samples 10
    python test_ollama_unified.py --model qwen2.5vl:3b --output custom_results --samples 100
    python test_ollama_unified.py --model rohithbojja/llava-med-v1.6:latest --samples 635
"""

import argparse
import requests
import json
import base64
import os
import re
import pandas as pd
import random
from PIL import Image
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
from create_multiple_choice import create_options, format_options, is_yes_no_question

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class UnifiedOllamaTester:
    def __init__(self, model_name, base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Create clean model identifier for results
        self.model_id = self._create_model_id(model_name)
    
    def _create_model_id(self, model_name):
        """Create a clean model identifier for file naming"""
        # Replace special characters and make it file-safe
        model_id = re.sub(r'[:/\-.]', '_', model_name)
        model_id = re.sub(r'_+', '_', model_id)  # Remove multiple underscores
        model_id = model_id.strip('_')  # Remove leading/trailing underscores
        return f"ollama_{model_id}"
    
    def encode_image_to_base64(self, image_path):
        """Convert image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return None
    
    def query_ollama(self, prompt, image_path=None):
        """Query Ollama API with text and optional image"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Add image if provided
            if image_path:
                image_b64 = self.encode_image_to_base64(image_path)
                if image_b64:
                    payload["images"] = [image_b64]
                else:
                    return "ERROR: Failed to encode image"
            
            print(f"Querying {self.model_name} with prompt: {prompt[:100]}...")
            
            response = requests.post(self.api_url, json=payload, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "ERROR: No response field")
            else:
                print(f"Error: HTTP {response.status_code}")
                return f"ERROR: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            print("Request timed out")
            return "ERROR: Request timeout"
        except Exception as e:
            print(f"Error querying Ollama: {str(e)}")
            return f"ERROR: {str(e)}"
    
    def process_single(self, image_path, question):
        """Process a single image and question"""
        try:
            print(f"\nProcessing image: {image_path}")
            print(f"Question: {question}")
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return "ERROR: Image not found"
            
            # Query the model
            response = self.query_ollama(question, image_path)
            print(f"Model response: {response}")
            return response
            
        except Exception as e:
            print(f"Unexpected error in process_single: {str(e)}")
            return "ERROR: Unexpected error occurred"
    
    def extract_option_from_response(self, response):
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
    
    def test_single_sample(self, sample, image_base_path=None):
        """Test a single sample with all sycophancy types"""
        try:
            print(f"\nTesting sample: {sample['id']}")
            if image_base_path:
                image_path = os.path.join(image_base_path, sample['image_path'])
            else:
                image_path = os.path.join("/home/zikun/github/Sycophency/Medical_Sycophency/3datasets/merged_images", sample['image_path'])
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
            initial_response = self.process_single(image_path, initial_question)
            initial_option = self.extract_option_from_response(initial_response)
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
                    sycophancy_response = self.process_single(image_path, sycophancy_question)
                    sycophancy_option = self.extract_option_from_response(sycophancy_response)
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
    
    def append_to_csv(self, df, filename):
        """Append a single row to CSV file"""
        try:
            if not os.path.exists(filename):
                df.to_csv(filename, index=False)
            else:
                df.to_csv(filename, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error writing to CSV: {str(e)}")
    
    def load_dataset(self):
        """Load dataset from CSV file"""
        csv_path = "sycophancy_challenge_set.csv"  # Changed to challenge set
        return pd.read_csv(csv_path)
    
    def test_samples(self, num_samples=5, output_filename=None, image_base_path=None):
        """Test model on samples from CSV"""
        try:
            print(f"Testing model: {self.model_name}")
            print("Loading dataset...")
            try:
                df = self.load_dataset()
                print(f"Loaded {len(df)} samples")
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
                return
            
            # Create results directory if it doesn't exist
            try:
                results_dir = "sycophancy_results"
                os.makedirs(results_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating results directory: {str(e)}")
                return
            
            # Use provided filename or generate default
            if output_filename is None:
                output_filename = f"{self.model_id}_results.csv"
            
            csv_filename = os.path.join(results_dir, output_filename)
            print(f"Results will be saved to: {csv_filename}")
            
            # Process samples
            if num_samples is None:
                samples_to_process = df
                print(f"\nProcessing ALL {len(df)} samples...")
            else:
                samples_to_process = df.head(num_samples)
                print(f"\nProcessing {len(samples_to_process)} samples...")
            
            for i, (_, sample) in enumerate(samples_to_process.iterrows(), 1):
                try:
                    print(f"\n{'='*50}")
                    print(f"Processing sample {i}/{len(samples_to_process)}: {sample['id']}")
                    print(f"{'='*50}")
                    
                    # Process single sample
                    test_result = self.test_single_sample(sample, image_base_path)
                    
                    result = {
                        'model': self.model_id,
                        'model_name': self.model_name,
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
                    df_result = pd.DataFrame([result])
                    self.append_to_csv(df_result, csv_filename)
                    print(f"Results written to CSV for sample {sample['id']}")
                    
                except Exception as e:
                    print(f"Error processing sample {sample['id']}: {str(e)}")
                    continue
            
            print(f"\n🎉 Testing completed! Results saved to: {csv_filename}")
                    
        except Exception as e:
            print(f"Unexpected error in test_samples: {str(e)}")
            return

def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(
        description="Unified Ollama Vision-Language Model Sycophancy Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_ollama_unified.py --model llava:7b --samples 10
  python test_ollama_unified.py --model qwen2.5vl:3b --output custom_results.csv --samples 100
  python test_ollama_unified.py --model rohithbojja/llava-med-v1.6:latest --samples 635
  python test_ollama_unified.py --model gemma3:12b --samples all
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Ollama model name (e.g., llava:7b, qwen2.5vl:3b, gemma3:12b)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=str,
        default='5',
        help='Number of samples to test (default: 5, use "all" for all 635 samples)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: <model_name>_results.csv)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama base URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--image-path',
        type=str,
        default='/home/zikun/github/Sycophency/Medical_Sycophency/3datasets/merged_images',
        help='Base path to image directory'
    )
    
    args = parser.parse_args()
    
    # Parse samples argument
    if args.samples.lower() == 'all':
        num_samples = None
    else:
        try:
            num_samples = int(args.samples)
        except ValueError:
            print(f"Error: Invalid samples value '{args.samples}'. Use a number or 'all'.")
            return
    
    # Download required NLTK data
    print("Downloading NLTK data...")
    nltk.download('wordnet', quiet=True)
    
    # Create tester instance
    print(f"Initializing tester for model: {args.model}")
    tester = UnifiedOllamaTester(args.model, args.base_url)
    
    # Run tests
    tester.test_samples(num_samples=num_samples, output_filename=args.output, image_base_path=args.image_path)

if __name__ == "__main__":
    main()
