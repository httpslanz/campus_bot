from django.contrib import admin
from .models import (Intent, TrainingData, ModelVersion, ChatLog, Office, OfficeUser, IntentResponse, Location, LocationKeyword, Feedback, Category)

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'order', 'created_at']
    search_fields = ['name', 'description']
    list_filter = ['order']
    ordering = ['order']

@admin.register(Intent)
class IntentAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'description', 'created_at']
    search_fields = ['name']

@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    list_display = ('intent', 'question_count', 'answer', 'is_active', 'is_reviewed')

    def question_count(self, obj):
        questions = obj.get_questions()
        return len(questions) if questions else 0

    question_count.short_description = "Questions"

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ['version', 'accuracy', 'training_samples', 'is_active', 'trained_at']
    list_filter = ['is_active']

@admin.register(ChatLog)
class ChatLogAdmin(admin.ModelAdmin):
    list_display = ['user_message', 'predicted_intent', 'confidence', 'timestamp']
    list_filter = ['predicted_intent', 'feedback']
    search_fields = ['user_message', 'bot_response']
    
@admin.register(Office)
class OfficeAdmin(admin.ModelAdmin):
    list_display = ['name', 'contact_email', 'is_active', 'created_at']
    search_fields = ['name', 'contact_email']
    list_filter = ['is_active']

@admin.register(OfficeUser)
class OfficeUserAdmin(admin.ModelAdmin):
    list_display = ['user', 'office', 'role', 'created_at']
    list_filter = ['office', 'role']
    search_fields = ['user__username', 'office__name']
    
@admin.register(IntentResponse)
class IntentResponseAdmin(admin.ModelAdmin):
    list_display = ['intent', 'answer', 'priority', 'is_default', 'is_active', 'updated_at']
    list_filter = ['intent', 'is_default', 'is_active']
    search_fields = ['answer']
    
@admin.register(Location)
class LocationAdmin(admin.ModelAdmin):
    list_display = ['room_number', 'room_name', 'building', 'floor', 'is_active']
    list_filter = ['building', 'floor', 'is_active']
    search_fields = ['room_number', 'room_name', 'building', 'description']
    
@admin.register(LocationKeyword)
class LocationKeywordAdmin(admin.ModelAdmin):
    list_display = ['keyword', 'location', 'priority']
    list_filter = ['location']
    search_fields = ['keyword']
    
@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ['feedback_type', 'get_rating_display', 'user', 'created_at', 'is_resolved']
    list_filter = ['feedback_type', 'is_resolved', 'created_at']
    search_fields = ['message', 'admin_notes', 'user__username']
    readonly_fields = ['created_at', 'ip_address']
    
    def get_rating_display(self, obj):
        if obj.rating:
            return '⭐' * obj.rating
        return '-'
    get_rating_display.short_description = 'Rating'