from django.core.management.base import BaseCommand
from chatbot.models import TrainingData
from chatbot.data_augmentation import BilingualAugmenter

class Command(BaseCommand):
    help = 'Augment training data locally (no API needed)'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--augmentations',
            type=int,
            default=4,
            help='Number of augmentations per question (default: 4)'
        )
        
        parser.add_argument(
            '--preview',
            action='store_true',
            help='Preview augmentations without saving'
        )
        
        parser.add_argument(
            '--intent',
            type=str,
            help='Augment specific intent only'
        )
    
    def handle(self, *args, **options):
        num_aug = options['augmentations']
        preview_only = options['preview']
        specific_intent = options['intent']
        
        augmenter = BilingualAugmenter()
        
        self.stdout.write("="*70)
        self.stdout.write("DATA AUGMENTATION (100% LOCAL)")
        self.stdout.write("="*70)
        
        # Filter training data
        if specific_intent:
            from chatbot.models import Intent
            try:
                intent_obj = Intent.objects.get(name=specific_intent)
                training_data = TrainingData.objects.filter(intent=intent_obj, is_active=True)
                self.stdout.write(f"\nAugmenting intent: {specific_intent}\n")
            except Intent.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"Intent '{specific_intent}' not found!"))
                return
        else:
            training_data = TrainingData.objects.filter(is_active=True)
        
        total_original = 0
        total_augmented = 0
        
        for training in training_data:
            original_questions = training.get_questions()
            
            if not original_questions:
                continue
            
            total_original += len(original_questions)
            all_questions = []
            
            self.stdout.write(f"\n📚 Intent: {training.intent.name}")
            if training.intent.category:
                self.stdout.write(f"   Category: {training.intent.category.name}")
            
            for i, question in enumerate(original_questions, 1):
                # Generate augmentations
                augmented = augmenter.augment_question(question, num_aug)
                all_questions.extend(augmented)
                
                if preview_only:
                    self.stdout.write(f"\n   {i}. Original: {question}")
                    for j, aug in enumerate(augmented[1:], 1):  # Skip original
                        self.stdout.write(f"      └─ Aug {j}: {aug}")
            
            # Remove duplicates
            all_questions = list(set(all_questions))
            total_augmented += len(all_questions)
            
            increase = len(all_questions) - len(original_questions)
            percentage = (increase / len(original_questions)) * 100 if len(original_questions) > 0 else 0
            
            status = f"{len(original_questions)} → {len(all_questions)} (+{increase}, +{percentage:.0f}%)"
            
            if not preview_only:
                # Save to database
                training.set_questions(all_questions)
                training.save()
                self.stdout.write(self.style.SUCCESS(f"   ✓ Saved: {status}"))
            else:
                self.stdout.write(f"   Preview: {status}")
        
        # Summary
        self.stdout.write("\n" + "="*70)
        self.stdout.write("SUMMARY:")
        self.stdout.write(f"  Original questions: {total_original}")
        self.stdout.write(f"  After augmentation: {total_augmented}")
        
        if total_original > 0:
            increase = total_augmented - total_original
            percentage = (increase / total_original) * 100
            self.stdout.write(f"  Increase: +{increase} questions (+{percentage:.1f}%)")
        
        if preview_only:
            self.stdout.write(self.style.WARNING("\n⚠️  PREVIEW MODE - No changes saved"))
            self.stdout.write("\nTo save changes, run:")
            self.stdout.write(f"  python manage.py augment_data --augmentations={num_aug}")
        else:
            self.stdout.write(self.style.SUCCESS("\n✓ AUGMENTATION COMPLETE!"))
            self.stdout.write("\nNext steps:")
            self.stdout.write("  1. Test for confusion: python test_confusion.py")
            self.stdout.write("  2. Retrain your model:")
            self.stdout.write("     python manage.py shell")
            self.stdout.write("     >>> from chatbot.ml_hybridpipeline import HybridChatbotPipeline")
            self.stdout.write("     >>> pipeline = HybridChatbotPipeline()")
            self.stdout.write("     >>> pipeline.train()")
        
        self.stdout.write("="*70)