import pandas as pd
import random
from nltk.corpus import wordnet
import nltk
import os

def is_yes_no_question(question, answer):
    """Check if this is a yes/no question"""
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Check if question starts with is/are/do/does
    if question_lower.startswith(('is ', 'are ', 'do ', 'does ')):
        return True
    
    # Check if answer is yes/no
    if answer_lower in ['yes', 'no', 'true', 'false']:
        return True
        
    return False

def get_synonyms_antonyms(word):
    """Get synonyms and antonyms for a word using WordNet"""
    synonyms = set()
    antonyms = set()
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name())
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    
    return list(synonyms), list(antonyms)

def create_options(question, correct_answer):
    """Create multiple choice options"""
    if is_yes_no_question(question, correct_answer):
        # For yes/no questions
        options = ['yes', 'no', 'not sure', 'cannot see clearly']
        # Map correct answer to yes/no
        correct_answer = 'yes' if correct_answer.lower() in ['yes', 'true'] else 'no'
    else:
        # For other questions
        # Get synonyms and antonyms
        synonyms, antonyms = get_synonyms_antonyms(correct_answer)
        
        # Create options pool
        options = [correct_answer]  # Start with correct answer
        
        # Add one synonym if available
        if synonyms:
            options.append(random.choice(synonyms))
        # Add one antonym if available
        if antonyms:
            options.append(random.choice(antonyms))
        
        # Ensure we have exactly 4 options
        while len(options) < 4:
            if len(options) == 1:
                options.append('not sure')
            elif len(options) == 2:
                options.append('cannot see clearly')
            else:
                # If we still need more options, add variations of "not sure"
                options.append('unable to determine')
    
    # Shuffle options
    random.shuffle(options)
    
    # Create option mapping
    option_mapping = {
        'A': options[0],
        'B': options[1],
        'C': options[2],
        'D': options[3]
    }
    
    # Find correct option
    correct_option = None
    for opt, ans in option_mapping.items():
        if ans == correct_answer:
            correct_option = opt
            break
    
    return option_mapping, correct_option

def format_options(option_mapping):
    """Format options for display"""
    return "\n".join([f"{opt}: {ans}" for opt, ans in option_mapping.items()])

def main():
    # Download required NLTK data
    nltk.download('wordnet')
    
    # Load dataset
    csv_path = os.path.join("challenge_dataset", "filtered_unique_choices.csv")
    df = pd.read_csv(csv_path)
    
    # Test a few samples
    for _, sample in df.head(5).iterrows():
        print("\n" + "="*50)
        print(f"Sample ID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Correct Answer: {sample['correct_answer']}")
        
        # Create options
        option_mapping, correct_option = create_options(
            sample['question'],
            sample['correct_answer']
        )
        
        # Print formatted options
        print("\nOptions:")
        print(format_options(option_mapping))
        print(f"\nCorrect Option: {correct_option}")

if __name__ == "__main__":
    main() 