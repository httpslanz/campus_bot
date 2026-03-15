from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
import json

# Create your models here.
class User(AbstractUser):
    pass

class Category(models.Model):
    """Categories for organizing intents in the chat menu"""
    name = models.CharField(max_length=100, unique=True, help_text="Category name (e.g., Enrollment, Scholarships)")
    slug = models.SlugField(max_length=100, unique=True, help_text="URL-friendly version")
    description = models.TextField(blank=True, help_text="Brief description of this category")
    order = models.IntegerField(default=0, help_text="Display order (lower numbers appear first)")
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='created_categories')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['order', 'name']
        verbose_name_plural = 'Categories'
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if not self.slug:
            from django.utils.text import slugify
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

class Intent(models.Model):
    name = models.CharField(max_length=150, unique=True)  # increased
    description = models.TextField(blank=True)
    category = models.ForeignKey(
        'Category', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='intents',
        help_text="Category for menu organization"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
class Office(models.Model):
    """Different campus offices that can submit tickets"""
    name = models.CharField(max_length=150)  # increased
    description = models.TextField(blank=True)
    contact_email = models.EmailField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']

class OfficeUser(models.Model):
    """Links users to offices"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='office_profile')
    office = models.ForeignKey(Office, on_delete=models.CASCADE, related_name='staff')
    role = models.CharField(max_length=50, choices=[
        ('staff', 'Staff'),
        ('manager', 'Manager'),
    ], default='staff')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.office.name}"

class TrainingData(models.Model):
    """Training data for chatbot intents"""
    intent = models.ForeignKey(Intent, on_delete=models.CASCADE, related_name='training_samples')
    questions_data = models.JSONField(default=list, help_text="List of sample questions")
    answer = models.TextField(help_text="Bot response for this intent")
    
    # NEW FIELDS FOR TRACKING
    submitted_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='submitted_training_data')
    submitted_at = models.DateTimeField(default=timezone.now)
    office = models.ForeignKey(Office, on_delete=models.SET_NULL, null=True, blank=True, related_name='training_data')
    is_reviewed = models.BooleanField(default=False, help_text="Has admin reviewed this?")
    reviewed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='reviewed_training_data')
    reviewed_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True, help_text="Submission notes from staff")
    
    # Existing fields
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        question_count = len(self.get_questions())
        return f"{self.intent.name} - {question_count} questions"
    
    def get_questions(self):
        """Returns list of questions"""
        return self.questions_data if self.questions_data else []
    
    def set_questions(self, questions_list):
        """Sets questions from a list"""
        self.questions_data = questions_list
    
    def mark_as_reviewed(self, admin_user):
        """Mark this training data as reviewed by admin"""
        from django.utils import timezone
        self.is_reviewed = True
        self.reviewed_by = admin_user
        self.reviewed_at = timezone.now()
        self.save()

class ModelVersion(models.Model):
    version = models.CharField(max_length=100)  # increased
    model_path = models.CharField(max_length=255)
    accuracy = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=False)
    trained_at = models.DateTimeField(auto_now_add=True)
    training_samples = models.IntegerField(default=0)
    
    def __str__(self):
        return f"v{self.version} - {self.accuracy:.2f}%" if self.accuracy else f"v{self.version}"

class ChatLog(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    predicted_intent = models.CharField(max_length=150, null=True)  # increased
    confidence = models.FloatField(null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    feedback = models.CharField(max_length=20, choices=[
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral')
    ], null=True, blank=True)

class IntentResponse(models.Model):
    """Store multiple possible responses for each intent"""
    intent = models.ForeignKey(Intent, on_delete=models.CASCADE, related_name='responses')
    answer = models.TextField()
    is_default = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    priority = models.IntegerField(default=0)  # Higher = shown first
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-priority', '-updated_at']
    
    def __str__(self):
        return f"{self.intent.name}: {self.answer[:50]}"
    

class Location(models.Model):
    """
    Stores information about campus locations (rooms, offices, facilities)
    """
    # Identifiers
    room_number = models.CharField(max_length=200, unique=True, help_text="e.g., 401, 402, Smart Lab")  # increased
    room_name = models.CharField(max_length=255, blank=True, help_text="e.g., Smart Lab, Computer Lab")
    
    # Location details
    building = models.CharField(max_length=200, help_text="e.g., Building A, Main Building")  # increased
    floor = models.CharField(max_length=50, help_text="e.g., 4th Floor, Ground Floor")  # increased
    
    # Description (optional)
    description = models.TextField(blank=True, help_text="Brief description of the room")
    
    # Aliases (alternative names)
    aliases = models.TextField(blank=True, help_text="JSON list of alternative names")
    
    # Status
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Location"
        verbose_name_plural = "Locations"
        ordering = ['building', 'floor', 'room_number']
    
    def get_aliases(self):
        """Get list of aliases"""
        try:
            return json.loads(self.aliases) if self.aliases else []
        except:
            return []
    
    def set_aliases(self, aliases_list):
        """Set aliases as JSON"""
        self.aliases = json.dumps(aliases_list, ensure_ascii=False)
    
    def __str__(self):
        return f"Room {self.room_number} ({self.building})"


class LocationKeyword(models.Model):
    """
    Keywords/patterns to help identify room mentions
    """
    location = models.ForeignKey(Location, on_delete=models.CASCADE, related_name='keywords')
    keyword = models.CharField(max_length=200, help_text="e.g., '401', 'smart lab', 'smartlab'")  # increased
    priority = models.IntegerField(default=1, help_text="Higher = more important")
    
    class Meta:
        unique_together = ['location', 'keyword']
    
    def __str__(self):
        return f"{self.keyword} → {self.location.room_number}"
    
    
class Feedback(models.Model):
    """
    Stores user feedback (ratings, suggestions, reports)
    """
    FEEDBACK_TYPES = [
        ('rating', 'Rating'),
        ('suggestion', 'Suggestion'),
        ('report', 'Report Issue'),
    ]
    
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPES)
    rating = models.IntegerField(null=True, blank=True, help_text="Star rating (1-5)")
    message = models.TextField(blank=True, help_text="Feedback message")
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_resolved = models.BooleanField(default=False)
    admin_notes = models.TextField(blank=True)
    
    class Meta:
        verbose_name = "Feedback"
        verbose_name_plural = "Feedbacks"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_feedback_type_display()} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_rating_display(self):
        if self.rating:
            return '⭐' * self.rating
        return '-'
    
class TrainingUpdateTicket(models.Model):

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    ticket_number = models.CharField(max_length=40, unique=True, editable=False)  # increased

    submitted_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='submitted_update_tickets')
    office = models.ForeignKey(Office, on_delete=models.CASCADE)

    training_data = models.ForeignKey(
        TrainingData, on_delete=models.CASCADE, related_name='update_tickets'
    )

    new_questions = models.TextField(help_text="Proposed additional questions")
    new_answer = models.TextField(blank=True, help_text="Proposed updated answer")
    reason = models.TextField(help_text="Why this update is needed")

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    admin_notes = models.TextField(blank=True)

    reviewed_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='reviewed_update_tickets'
    )

    reviewed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)


    def save(self, *args, **kwargs):

        if not self.ticket_number:

            today = timezone.now().strftime('%Y%m%d')

            count = TrainingUpdateTicket.objects.filter(
                ticket_number__startswith=f'UPD-{today}'
            ).count()

            self.ticket_number = f'UPD-{today}-{count+1:03d}'

        super().save(*args, **kwargs)


    def __str__(self):
        return f"{self.ticket_number} - {self.training_data.intent.name}"