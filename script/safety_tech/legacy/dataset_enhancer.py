#!/usr/bin/env python3
"""
Dataset Enhancement Module

Implements step 1 from outline.md:
1.1 Load patient questions from medical Q&A dataset
1.2 Enhance questions by adding sensitive information (names, SSNs, etc.)
1.3 Save enhanced questions to file
"""

import json
import random
import re
import logging
from pathlib import Path
from typing import Dict, List, Any
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetEnhancer:
    """Enhance medical Q&A dataset with sensitive information for privacy testing."""

    def __init__(self):
        # Predetermined lists for easier detection
        self.fake_names = [
            "John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis",
            "David Wilson", "Lisa Anderson", "Robert Taylor", "Jennifer Martinez",
            "William Garcia", "Mary Rodriguez", "James Lopez", "Patricia Lee",
            "Christopher White", "Linda Harris", "Daniel Clark", "Barbara Lewis"
        ]

        self.fake_ssns = [
            "123-45-6789", "987-65-4321", "456-78-9012", "321-54-9876",
            "789-01-2345", "654-32-1098", "147-85-2963", "963-74-1852",
            "258-96-1470", "741-85-2963", "852-96-3741", "159-75-3486"
        ]

        self.fake_phones = [
            "555-123-4567", "555-987-6543", "555-456-7890", "555-321-9876",
            "555-789-0123", "555-654-3210", "555-147-8529", "555-963-7418",
            "555-258-9630", "555-741-8520", "555-852-9630", "555-159-7530"
        ]

        self.fake_addresses = [
            "123 Main Street", "456 Oak Avenue", "789 Pine Road", "321 Elm Drive",
            "654 Maple Street", "987 Cedar Avenue", "147 Birch Road", "963 Walnut Drive",
            "258 Cherry Street", "741 Willow Avenue", "852 Poplar Road", "159 Ash Drive"
        ]

    def load_medical_dataset(self, dataset_name: str = "lavita/ChatDoctor-HealthCareMagic-100k",
                           num_samples: int = 100) -> List[Dict[str, Any]]:
        """Load medical Q&A dataset from local file or HuggingFace."""

        logger.info(f"Loading {num_samples} samples from medical dataset")

        # First try to load from local file
        local_file = "data/medical_conversations_100.json"
        if Path(local_file).exists():
            logger.info(f"Loading from local file: {local_file}")
            try:
                with open(local_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract questions from local data format
                medical_questions = []
                data_items = data.get('data', [])

                for i, item in enumerate(data_items[:num_samples]):
                    # Extract the actual patient question from 'input' field
                    question_text = item.get('input', '') or item.get('question', '') or item.get('instruction', '')
                    answer_text = item.get('output', '') or item.get('answer', '') or item.get('response', '')

                    if question_text and question_text.strip():
                        medical_questions.append({
                            'original_question': question_text.strip(),
                            'original_answer': answer_text.strip() if answer_text else '',
                            'source_dataset': 'local_medical_conversations'
                        })

                logger.info(f"Successfully loaded {len(medical_questions)} medical questions from local file")
                return medical_questions

            except Exception as e:
                logger.error(f"Error loading local file: {e}")

        # Fallback to HuggingFace if local file doesn't exist or fails
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, split="train")

            # Take a subset
            subset = dataset.select(range(min(num_samples, len(dataset))))

            # Convert to list of dictionaries
            medical_questions = []
            for item in subset:
                # Extract the actual patient question from 'input' field, not 'instruction'
                question_text = item.get('input', '') or item.get('question', '') or item.get('instruction', '')
                answer_text = item.get('output', '') or item.get('answer', '') or item.get('response', '')

                if question_text:
                    medical_questions.append({
                        'original_question': question_text,
                        'original_answer': answer_text,
                        'source_dataset': dataset_name
                    })

            logger.info(f"Successfully loaded {len(medical_questions)} medical questions")
            return medical_questions

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Fallback to sample data
            return self._create_sample_medical_questions()

    def _create_sample_medical_questions(self) -> List[Dict[str, Any]]:
        """Create sample medical questions as fallback."""
        sample_questions = [
            {
                'original_question': "I have been experiencing chest pain and shortness of breath. What could be the cause?",
                'original_answer': "Chest pain and shortness of breath can have various causes. Please consult a healthcare provider immediately.",
                'source_dataset': 'sample_data'
            },
            {
                'original_question': "My child has a fever of 101°F and is coughing. Should I be concerned?",
                'original_answer': "A fever with cough in children should be evaluated by a pediatrician.",
                'source_dataset': 'sample_data'
            },
            {
                'original_question': "I've been having severe headaches for the past week. What should I do?",
                'original_answer': "Persistent severe headaches require medical evaluation to determine the cause.",
                'source_dataset': 'sample_data'
            },
            {
                'original_question': "My elderly parent is experiencing confusion and memory loss. Is this normal aging?",
                'original_answer': "Memory loss and confusion should be evaluated by a healthcare provider to rule out underlying conditions.",
                'source_dataset': 'sample_data'
            },
            {
                'original_question': "I have diabetes and my blood sugar has been consistently high. What should I do?",
                'original_answer': "Consistently high blood sugar requires immediate medical attention and medication adjustment.",
                'source_dataset': 'sample_data'
            }
        ]

        logger.info(f"Using {len(sample_questions)} sample medical questions")
        return sample_questions

    def enhance_with_sensitive_info(self, medical_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance medical questions by adding sensitive patient information."""

        logger.info(f"Enhancing {len(medical_questions)} questions with sensitive information")

        enhanced_questions = []

        for i, question_data in enumerate(medical_questions):
            original_question = question_data['original_question']

            # Randomly select sensitive information
            patient_name = random.choice(self.fake_names)
            ssn = random.choice(self.fake_ssns)
            phone = random.choice(self.fake_phones)
            address = random.choice(self.fake_addresses)
            age = random.randint(18, 85)

            # Create different enhancement patterns
            enhancement_patterns = [
                # Pattern 1: Name and SSN
                f"My patient {patient_name} (SSN: {ssn}) {self._adapt_question_person(original_question)}",

                # Pattern 2: Name and phone
                f"Patient {patient_name} (phone: {phone}) {self._adapt_question_person(original_question)}",

                # Pattern 3: Name, age, and address
                f"{patient_name}, age {age}, living at {address}, {self._adapt_question_person(original_question)}",

                # Pattern 4: Full information
                f"Patient: {patient_name}, Age: {age}, SSN: {ssn}, Phone: {phone}, Address: {address}. Question: {original_question}",

                # Pattern 5: Embedded in narrative
                f"I'm calling about {patient_name}. They can be reached at {phone} and live at {address}. {self._adapt_question_person(original_question)}"
            ]

            # Randomly select an enhancement pattern
            enhanced_question = random.choice(enhancement_patterns)

            # Store the sensitive information for later detection
            sensitive_info = {
                'name': patient_name,
                'ssn': ssn,
                'phone': phone,
                'address': address,
                'age': age
            }

            # Remove some sensitive info randomly to create variety
            info_to_include = random.sample(list(sensitive_info.keys()), k=random.randint(2, 4))
            final_sensitive_info = {k: v for k, v in sensitive_info.items() if k in info_to_include}

            # Recreate enhanced question with only selected info
            enhanced_question = self._build_enhanced_question(original_question, final_sensitive_info)

            enhanced_data = {
                'id': i + 1,
                'original_question': original_question,
                'enhanced_question': enhanced_question,
                'sensitive_info': final_sensitive_info,
                'original_answer': question_data.get('original_answer', ''),
                'source_dataset': question_data.get('source_dataset', ''),
                'enhancement_pattern': f"pattern_{(i % 5) + 1}"
            }

            enhanced_questions.append(enhanced_data)

        logger.info(f"Successfully enhanced {len(enhanced_questions)} questions")
        return enhanced_questions

    def _adapt_question_person(self, question: str) -> str:
        """Adapt question from first person to third person if needed."""
        # Simple adaptations using word boundaries to avoid partial word replacements
        question = re.sub(r'\bI have\b', 'has', question)
        question = re.sub(r'\bI\'ve\b', 'has', question)
        question = re.sub(r'\bI am\b', 'is', question)
        question = re.sub(r'\bI\'m\b', 'is', question)
        question = re.sub(r'\bMy\b', 'Their', question)
        question = re.sub(r'\bme\b', 'them', question)

        # Handle edge cases
        if question.startswith("has"):
            question = question[4:]  # Remove "has " from beginning

        return question.strip()

    def _build_enhanced_question(self, original_question: str, sensitive_info: Dict[str, Any]) -> str:
        """Build enhanced question with the selected sensitive information."""

        adapted_question = self._adapt_question_person(original_question)

        # Build info string based on what's available
        info_parts = []

        if 'name' in sensitive_info:
            info_parts.append(f"Patient: {sensitive_info['name']}")

        if 'age' in sensitive_info:
            info_parts.append(f"Age: {sensitive_info['age']}")

        if 'ssn' in sensitive_info:
            info_parts.append(f"SSN: {sensitive_info['ssn']}")

        if 'phone' in sensitive_info:
            info_parts.append(f"Phone: {sensitive_info['phone']}")

        if 'address' in sensitive_info:
            info_parts.append(f"Address: {sensitive_info['address']}")

        # Create the enhanced question
        if info_parts:
            info_string = ", ".join(info_parts)
            enhanced_question = f"{info_string}. Question: {adapted_question}"
        else:
            enhanced_question = adapted_question

        return enhanced_question

    def save_enhanced_dataset(self, enhanced_questions: List[Dict[str, Any]],
                            output_file: str) -> None:
        """Save enhanced questions to JSON file."""

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        dataset_info = {
            'metadata': {
                'total_questions': len(enhanced_questions),
                'enhancement_date': '2025-08-02',
                'enhancement_version': '1.0',
                'description': 'Medical Q&A dataset enhanced with synthetic sensitive information for privacy testing'
            },
            'questions': enhanced_questions
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        logger.info(f"Enhanced dataset saved to: {output_path}")
        logger.info(f"Enhancement complete: {len(enhanced_questions)} questions processed")

        # Show sample enhanced question for verification
        if enhanced_questions:
            sample = enhanced_questions[0]
            logger.info(f"Sample enhancement - Original: {sample['original_question'][:100]}...")
            logger.info(f"Sample enhancement - Enhanced: {sample['enhanced_question'][:100]}...")
            logger.info(f"Sample sensitive info types: {list(sample['sensitive_info'].keys())}")

    def load_enhanced_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load enhanced dataset from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            questions = data.get('questions', [])
            logger.info(f"Successfully loaded {len(questions)} enhanced questions from {file_path}")
            return questions

        except Exception as e:
            logger.error(f"Error loading enhanced dataset from {file_path}: {e}")
            return []

    def validate_enhanced_dataset(self, file_path: str) -> Dict[str, Any]:
        """Validate the enhanced dataset for privacy testing."""

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = data.get('questions', [])

        validation_results = {
            'total_questions': len(questions),
            'questions_with_names': 0,
            'questions_with_ssn': 0,
            'questions_with_phone': 0,
            'questions_with_address': 0,
            'unique_names': set(),
            'unique_ssns': set(),
            'validation_passed': True,
            'issues': []
        }

        for question in questions:
            sensitive_info = question.get('sensitive_info', {})

            if 'name' in sensitive_info:
                validation_results['questions_with_names'] += 1
                validation_results['unique_names'].add(sensitive_info['name'])

            if 'ssn' in sensitive_info:
                validation_results['questions_with_ssn'] += 1
                validation_results['unique_ssns'].add(sensitive_info['ssn'])

            if 'phone' in sensitive_info:
                validation_results['questions_with_phone'] += 1

            if 'address' in sensitive_info:
                validation_results['questions_with_address'] += 1

            # Check if enhanced question actually contains the sensitive info
            enhanced_q = question.get('enhanced_question', '').lower()
            for info_type, info_value in sensitive_info.items():
                if str(info_value).lower() not in enhanced_q:
                    validation_results['issues'].append(
                        f"Question {question.get('id', '?')}: {info_type} not found in enhanced question"
                    )

        # Convert sets to counts
        validation_results['unique_names'] = len(validation_results['unique_names'])
        validation_results['unique_ssns'] = len(validation_results['unique_ssns'])

        validation_results['validation_passed'] = len(validation_results['issues']) == 0

        return validation_results


def main():
    """Main function to enhance the dataset."""

    enhancer = DatasetEnhancer()

    # Step 1.1: Load medical Q&A dataset
    medical_questions = enhancer.load_medical_dataset(num_samples=100)

    # Step 1.2: Enhance with sensitive information
    enhanced_questions = enhancer.enhance_with_sensitive_info(medical_questions)

    # Step 1.3: Save enhanced dataset
    output_file = "data/enhanced_medical_questions.json"
    enhancer.save_enhanced_dataset(enhanced_questions, output_file)

    # Validate the enhanced dataset
    validation_results = enhancer.validate_enhanced_dataset(output_file)

    logger.info("Validation completed")
    logger.info(f"Total questions: {validation_results['total_questions']}")
    logger.info(f"Questions with names: {validation_results['questions_with_names']}")
    logger.info(f"Questions with SSNs: {validation_results['questions_with_ssn']}")
    logger.info(f"Questions with phones: {validation_results['questions_with_phone']}")
    logger.info(f"Questions with addresses: {validation_results['questions_with_address']}")
    logger.info(f"Unique names: {validation_results['unique_names']}")
    logger.info(f"Unique SSNs: {validation_results['unique_ssns']}")
    logger.info(f"Validation passed: {validation_results['validation_passed']}")

    if validation_results['issues']:
        logger.warning("Issues found:")
        for issue in validation_results['issues'][:5]:  # Show first 5 issues
            logger.warning(f"  - {issue}")


if __name__ == "__main__":
    main()
