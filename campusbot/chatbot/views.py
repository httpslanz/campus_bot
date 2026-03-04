from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from .ml_pipeline import ChatbotMLPipeline
from .predictor import ChatbotPredictor
from .ml_hybridpipeline import HybridChatbotPipeline
from .hybrid_predictor import HybridChatbotPredictor
import json
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout, authenticate
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.db.models import Count, Avg, Q
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from datetime import datetime, timedelta
from .models import (Intent, TrainingData, DataTicket, Office, IntentResponse, OfficeUser, TicketComment, ModelVersion, ChatLog, Location, LocationKeyword, Feedback, User, LocationTicket, LocationTicketComment, TrainingUpdateTicket, Category)
from django.utils.text import slugify


@login_required
def admin_dashboard(request):
    """Admin dashboard to manage training data"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')
    
    intents = Intent.objects.all()
    models = ModelVersion.objects.all().order_by('-trained_at')[:10]
    categories = Category.objects.filter(is_active=True)
    
    # Basic stats
    stats = {
        'total_intents': Intent.objects.count(),
        'total_training_data': TrainingData.objects.filter(is_active=True).count(),
        'total_chats': ChatLog.objects.count(),
        'pending_tickets': DataTicket.objects.filter(status='pending').count() if 'DataTicket' in dir() else 0,
    }
    
    # NEW: Unreviewed training data count
    unreviewed_count = TrainingData.objects.filter(
        is_reviewed=False,
        is_active=True,
        submitted_by__isnull=False
    ).count()
    
    # NEW: Recent submissions (last 10, prioritize unreviewed)
    recent_submissions = TrainingData.objects.filter(
        is_active=True,
        submitted_by__isnull=False
    ).select_related('intent', 'submitted_by', 'office').order_by(
        'is_reviewed',  # Unreviewed first
        '-submitted_at'
    )[:10]
    
    # Analytics
    thirty_days_ago = timezone.now() - timedelta(days=30)
    seven_days_ago = timezone.now() - timedelta(days=7)
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Conversation analytics
    total_conversations = ChatLog.objects.count()
    successful_count = ChatLog.objects.filter(predicted_intent__isnull=False).count()
    successful_rate = round((successful_count / total_conversations * 100) if total_conversations > 0 else 0, 1)
    
    avg_confidence = ChatLog.objects.filter(
        confidence__isnull=False
    ).aggregate(Avg('confidence'))['confidence__avg']
    avg_confidence = round(avg_confidence, 1) if avg_confidence else 0
    
    today_count = ChatLog.objects.filter(timestamp__gte=today_start).count()
    
    # Confidence distribution
    high_confidence = ChatLog.objects.filter(confidence__gte=80).count()
    medium_confidence = ChatLog.objects.filter(confidence__gte=50, confidence__lt=80).count()
    low_confidence = ChatLog.objects.filter(confidence__lt=50, confidence__isnull=False).count()
    
    total_with_conf = high_confidence + medium_confidence + low_confidence
    high_percent = round((high_confidence / total_with_conf * 100) if total_with_conf > 0 else 0, 1)
    medium_percent = round((medium_confidence / total_with_conf * 100) if total_with_conf > 0 else 0, 1)
    low_percent = round((low_confidence / total_with_conf * 100) if total_with_conf > 0 else 0, 1)
    
    # Intent distribution for chart (last 30 days)
    intent_distribution = ChatLog.objects.filter(
        timestamp__gte=thirty_days_ago,
        predicted_intent__isnull=False
    ).values('predicted_intent').annotate(
        count=Count('id')
    ).order_by('-count')[:10]
    
    intent_chart_data = {
        'labels': [item['predicted_intent'] for item in intent_distribution],
        'data': [item['count'] for item in intent_distribution]
    }
    
    # Trend data (last 7 days)
    trend_data = []
    labels = []
    for i in range(6, -1, -1):
        day = timezone.now() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        count = ChatLog.objects.filter(timestamp__gte=day_start, timestamp__lt=day_end).count()
        trend_data.append(count)
        labels.append(day.strftime('%a'))
    
    trend_chart_data = {
        'labels': labels,
        'data': trend_data
    }
    
    analytics = {
        'successful_count': successful_count,
        'successful_rate': successful_rate,
        'total_conversations': total_conversations,
        'avg_confidence': avg_confidence,
        'today_count': today_count,
        'high_confidence_count': high_confidence,
        'medium_confidence_count': medium_confidence,
        'low_confidence_count': low_confidence,
        'high_confidence_percent': high_percent,
        'medium_confidence_percent': medium_percent,
        'low_confidence_percent': low_percent,
        'intent_chart_data': intent_chart_data,
        'trend_chart_data': trend_chart_data,
    }
    
    # Check for pending location tickets
    try:
        from .models import LocationTicket
        pending_location_tickets = LocationTicket.objects.filter(status='pending').count()
    except:
        pending_location_tickets = 0
    
    context = {
        'categories': categories,
        'intents': intents,
        'models': models,
        'stats': stats,
        'analytics': analytics,
        'unreviewed_count': unreviewed_count,
        'recent_submissions': recent_submissions,
        'pending_tickets': stats.get('pending_tickets', 0),
        'pending_location_tickets': pending_location_tickets,
    }
    
    context['analytics']['intent_chart_data'] = json.dumps(intent_chart_data)
    context['analytics']['trend_chart_data'] = json.dumps(trend_chart_data)
    
    return render(request, 'admin_dashboard.html', context)

@login_required
def get_training_data_ajax(request):
    """AJAX endpoint to get paginated and filtered training data"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    # Get parameters
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 25))
    search = request.GET.get('search', '').strip()
    intent_id = request.GET.get('intent', '')
    status = request.GET.get('status', '')
    
    # Base queryset
    queryset = TrainingData.objects.select_related('intent').all()
    
    # Apply filters
    if search:
        # Search in both questions JSON and answer
        queryset = queryset.filter(
            Q(questions__icontains=search) | Q(answer__icontains=search)
        )
    
    if intent_id:
        queryset = queryset.filter(intent_id=intent_id)
    
    if status == 'active':
        queryset = queryset.filter(is_active=True)
    elif status == 'inactive':
        queryset = queryset.filter(is_active=False)
    
    # Order by most recent
    queryset = queryset.order_by('-created_at')
    
    # Paginate
    paginator = Paginator(queryset, per_page)
    page_obj = paginator.get_page(page)
    
    # Prepare results
    results = []
    for item in page_obj:
        questions = item.get_questions()
        # Show first question as preview
        first_question = questions[0] if questions else "No questions"
        question_preview = f"{first_question} (+{len(questions)-1} more)" if len(questions) > 1 else first_question
        
        results.append({
            'id': item.id,
            'intent_name': item.intent.name,
            'question': question_preview,  # Preview of questions
            'questions_count': len(questions),
            'answer': item.answer,
            'is_active': item.is_active,
            'created_at': item.created_at.isoformat(),
        })
    
    return JsonResponse({
        'results': results,
        'pagination': {
            'current_page': page_obj.number,
            'total_pages': paginator.num_pages,
            'total': paginator.count,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
        }
    })


@login_required
def get_conversations_ajax(request):
    """AJAX endpoint to get paginated conversation logs"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    # Get parameters
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 25))
    search = request.GET.get('search', '').strip()
    intent_filter = request.GET.get('intent', '').strip()
    confidence_filter = request.GET.get('confidence', '').strip()
    
    # Base queryset
    queryset = ChatLog.objects.all()
    
    # Apply filters
    if search:
        queryset = queryset.filter(
            Q(user_message__icontains=search) | Q(bot_response__icontains=search)
        )
    
    if intent_filter:
        queryset = queryset.filter(predicted_intent=intent_filter)
    
    if confidence_filter == 'high':
        queryset = queryset.filter(confidence__gte=80)
    elif confidence_filter == 'medium':
        queryset = queryset.filter(confidence__gte=50, confidence__lt=80)
    elif confidence_filter == 'low':
        queryset = queryset.filter(confidence__lt=50, confidence__isnull=False)
    elif confidence_filter == 'none':
        queryset = queryset.filter(predicted_intent__isnull=True)
    
    # Order by most recent
    queryset = queryset.order_by('-timestamp')
    
    # Paginate
    paginator = Paginator(queryset, per_page)
    page_obj = paginator.get_page(page)
    
    # Prepare results
    results = []
    for item in page_obj:
        results.append({
            'id': item.id,
            'user_message': item.user_message,
            'bot_response': item.bot_response,
            'predicted_intent': item.predicted_intent,
            'confidence': item.confidence,
            'timestamp': item.timestamp.isoformat(),
        })
    
    return JsonResponse({
        'results': results,
        'pagination': {
            'current_page': page_obj.number,
            'total_pages': paginator.num_pages,
            'total': paginator.count,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
        }
    })
    

# CREATE
@login_required
@require_http_methods(["POST"])
def create_training_data_ajax(request):
    """AJAX endpoint to create new training data"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        data = json.loads(request.body)
        intent_id = data.get('intent_id')
        questions = data.get('questions', [])
        answer = data.get('answer', '')
        
        # Validation
        if not intent_id:
            return JsonResponse({'success': False, 'error': 'Intent is required'})
        
        if not questions or len(questions) == 0:
            return JsonResponse({'success': False, 'error': 'At least one question is required'})
        
        if not answer:
            return JsonResponse({'success': False, 'error': 'Answer is required'})
        
        intent = Intent.objects.get(id=intent_id)
        
        # Check if training data for this intent exists
        existing = TrainingData.objects.filter(intent=intent, is_active=True).first()
        
        if existing:
            # Add to existing
            current_questions = existing.get_questions()
            new_questions = [q for q in questions if q not in current_questions]
            
            if new_questions:
                current_questions.extend(new_questions)
                existing.set_questions(current_questions)
                existing.answer = answer
                existing.save()
                
                return JsonResponse({
                    'success': True,
                    'message': f'Added {len(new_questions)} new question(s)',
                    'data': {
                        'id': existing.id,
                        'questions_count': len(current_questions)
                    }
                })
            else:
                return JsonResponse({'success': False, 'error': 'All questions already exist'})
        else:
            # Create new
            training = TrainingData.objects.create(
                intent=intent,
                answer=answer,
                is_active=True
            )
            training.set_questions(questions)
            
            return JsonResponse({
                'success': True,
                'message': 'Training data created successfully',
                'data': {
                    'id': training.id,
                    'questions_count': len(questions)
                }
            })
        
    except Intent.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Intent not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# READ (Detail)
@login_required
def get_training_data_detail_ajax(request, training_id):
    """AJAX endpoint to get training data details"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        # ✅ ALSO load category
        training = TrainingData.objects.select_related('intent__category').get(id=training_id)
        
        return JsonResponse({
            'success': True,
            'data': {
                'id': training.id,
                'intent_id': training.intent.id,
                'intent_name': training.intent.name,
                'intent_description': training.intent.description,

                # ✅ ADD THIS (THIS IS THE FIX)
                'category_id': training.intent.category.id if training.intent.category else None,
                'category_name': training.intent.category.name if training.intent.category else None,

                'questions': training.get_questions(),
                'answer': training.answer,
                'is_active': training.is_active,
                'created_at': training.created_at.isoformat(),
                'updated_at': training.updated_at.isoformat(),
            }
        })
    except TrainingData.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Training data not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# UPDATE
@login_required
@require_http_methods(["PUT"])
def update_training_data_ajax(request, training_id):

    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    try:
        data = json.loads(request.body)

        training = TrainingData.objects.select_related('intent').get(id=training_id)

        category_id = data.get('category_id')
        intent_id = data.get('intent_id')
        intent_description = data.get('intent_description')  # ✅ NEW
        questions = data.get('questions', [])
        answer = data.get('answer', '')
        is_active = data.get('is_active', True)

        # ================= VALIDATION =================

        if not intent_id:
            return JsonResponse({'success': False, 'error': 'Intent is required'})

        if not questions:
            return JsonResponse({'success': False, 'error': 'At least one question is required'})

        if not answer:
            return JsonResponse({'success': False, 'error': 'Answer is required'})

        # ================= UPDATE =================

        intent = Intent.objects.get(id=intent_id)
        
        # ✅ Update category
        if category_id:
            intent.category_id = category_id
        else:
            intent.category = None

        # ✅ Update description
        if intent_description is not None:
            intent.description = intent_description
            
        intent.save()

        training.intent = intent
        training.set_questions(questions)
        training.answer = answer
        training.is_active = is_active

        training.save()

        return JsonResponse({
            'success': True,
            'message': 'Training data updated successfully',
            'data': {
                'id': training.id,
                'intent_id': intent.id,
                'intent_name': intent.name,
                'intent_description': intent.description,  # ✅ RETURN
                'category_id': training.intent.category.id if training.intent.category else None,
                'category_name': training.intent.category.name if training.intent.category else None,
                'questions': training.get_questions(),
                'answer': training.answer,
                'is_active': training.is_active,
                'created_at': training.created_at.isoformat(),
                'updated_at': training.updated_at.isoformat(),
            }
        })

    except TrainingData.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Training data not found'})

    except Intent.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Intent not found'})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


# DELETE
@login_required
@require_http_methods(["DELETE"])
def delete_training_data_ajax(request, training_id):
    """AJAX endpoint to delete training data"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        training = TrainingData.objects.get(id=training_id)
        intent_name = training.intent.name
        questions_count = len(training.get_questions())
        
        training.delete()
        
        return JsonResponse({
            'success': True,
            'message': f'Deleted training data for {intent_name} ({questions_count} questions)'
        })
        
    except TrainingData.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Training data not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
def manage_categories(request):
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')

    # Search parameters
    category_query = request.GET.get('category_search', '')
    intent_query = request.GET.get('intent_search', '')

    categories = Category.objects.all()
    intents = Intent.objects.select_related('category').all()

    # Filtering
    if category_query:
        categories = categories.filter(
            Q(name__icontains=category_query) |
            Q(description__icontains=category_query)
        )

    if intent_query:
        intents = intents.filter(
            Q(name__icontains=intent_query) |
            Q(description__icontains=intent_query)
        )

    # Pagination
    category_paginator = Paginator(categories, 5)  # 5 per page
    intent_paginator = Paginator(intents, 5)

    category_page_number = request.GET.get('category_page')
    intent_page_number = request.GET.get('intent_page')

    category_page_obj = category_paginator.get_page(category_page_number)
    intent_page_obj = intent_paginator.get_page(intent_page_number)

    return render(request, 'manage_categories.html', {
        'categories': category_page_obj,
        'intents': intent_page_obj,
        'category_query': category_query,
        'intent_query': intent_query,
    })
    
@login_required
@require_http_methods(["POST"])
def update_category(request, category_id):
    """Update category"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})
    
    try:
        category = Category.objects.get(id=category_id)
        data = json.loads(request.body)
        
        category.name = data.get('name')
        category.icon = data.get('icon')
        category.description = data.get('description', '')
        category.order = int(data.get('order', 0))
        category.is_active = data.get('is_active', True)
        category.save()
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
@login_required
@require_http_methods(["POST"])
def delete_category(request, category_id):
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})

    try:
        category = Category.objects.get(id=category_id)
        category.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
@login_required
@require_http_methods(["POST"])
def create_intent(request):
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})

    try:
        data = json.loads(request.body)

        intent = Intent.objects.create(
            name=data.get('name'),
            description=data.get('description', ''),
            category_id=data.get('category_id')
        )

        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
@login_required
@require_http_methods(["POST"])
def update_intent(request, intent_id):
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})

    try:
        intent = Intent.objects.get(id=intent_id)
        data = json.loads(request.body)

        intent.name = data.get('name')
        intent.description = data.get('description', '')
        intent.category_id = data.get('category_id')
        intent.save()

        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
@login_required
@require_http_methods(["POST"])
def delete_intent(request, intent_id):
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})

    try:
        intent = Intent.objects.get(id=intent_id)
        intent.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@require_http_methods(["GET"])
def get_menu_categories(request):
    """Get dynamic menu categories from database"""
    try:
        # Get regular categories with intents
        categories = Category.objects.filter(is_active=True).prefetch_related(
            'intents__training_samples'
        ).order_by('order', 'name')
        
        formatted_categories = {}
        
        # Add regular intent-based categories
        for category in categories:
            intents_list = []
            
            for intent in category.intents.all():
                training = TrainingData.objects.filter(intent=intent, is_active=True).first()
                if training:
                    questions = training.get_questions()
                    example_question = questions[0] if questions else intent.description
                    
                    intents_list.append({
                        'intent': intent.name,
                        'description': intent.description,
                        'example_question': example_question,
                        'total_questions': len(questions)
                    })
            
            if intents_list:
                formatted_categories[category.slug] = {
                    'label': category.name,
                    'description': category.description,
                    'intents': intents_list,
                    'type': 'intent'  # Regular intent category
                }
        
        # Add special LOCATIONS category
        locations = Location.objects.filter(is_active=True).prefetch_related('keywords').order_by('building', 'floor', 'room_number')
        
        if locations.exists():
            location_items = []
            
            for location in locations:
                # Get first keyword as example
                first_keyword = location.keywords.first()
                example_query = f"Where is {first_keyword.keyword}?" if first_keyword else f"Where is {location.room_number}?"
                
                location_name = location.room_name if location.room_name else f"Room {location.room_number}"
                
                location_items.append({
                    'intent': f'location_{location.id}',
                    'description': location_name,
                    'example_question': example_query,
                    'room_number': location.room_number,
                    'building': location.building,
                    'floor': location.floor
                })
            
            formatted_categories['find-locations'] = {
                'label': 'Find a Room',
                'description': 'Search for campus locations and rooms',
                'intents': location_items,
                'type': 'location'  # Special location category
            }
        
        return JsonResponse({
            'success': True,
            'categories': formatted_categories
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)



@login_required
def add_training_data(request):
    """Admin add training data - same approach as office staff"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')
    
    if request.method == 'POST':
        try:
            entry_count = 0
            created_entries = []
            index = 1
            
            # Loop through entries
            while f'answer_{index}' in request.POST:
                category_id = request.POST.get(f'category_{index}')
                new_category_name = request.POST.get(f'new_category_name_{index}', '').strip()
                new_category_description = request.POST.get(f'new_category_description_{index}', '').strip()
                
                intent_id = request.POST.get(f'intent_{index}')
                new_intent_name = request.POST.get(f'new_intent_name_{index}', '').strip()
                new_intent_description = request.POST.get(f'new_intent_description_{index}', '').strip()
                answer = request.POST.get(f'answer_{index}', '').strip()
                
                # Get questions array
                questions_list = request.POST.getlist(f'questions_{index}[]')
                questions_list = [q.strip() for q in questions_list if q.strip()]
                
                # Validation
                if not questions_list:
                    messages.error(request, f'Entry {index}: Please provide at least one question.')
                    return redirect('add_training_data')
                
                if not answer:
                    messages.error(request, f'Entry {index}: Answer is required.')
                    return redirect('add_training_data')
                
                # Handle category (existing or new)
                category = None
                if category_id == '__new__' and new_category_name:
                    # Create new category
                    category, created = Category.objects.get_or_create(
                        name=new_category_name,
                        defaults={
                            'description': new_category_description,
                            'created_by': request.user,
                            'is_active': True
                        }
                    )
                    
                    if created:
                        messages.info(request, f'✓ New category "{category.name}" created!')
                    
                elif category_id:
                    try:
                        category = Category.objects.get(id=category_id)
                    except Category.DoesNotExist:
                        messages.error(request, f'Entry {index}: Invalid category.')
                        return redirect('add_training_data')
                else:
                    messages.error(request, f'Entry {index}: Please select or create a category.')
                    return redirect('add_training_data')
                
                # Get or create intent
                intent = None
                if new_intent_name:
                    intent, created = Intent.objects.get_or_create(
                        name=new_intent_name,
                        defaults={
                            'description': new_intent_description,
                            'category': category
                        }
                    )
                    if not created and intent.category != category:
                        # Update category if intent exists
                        intent.category = category
                        intent.save()
                elif intent_id:
                    try:
                        intent = Intent.objects.get(id=intent_id)
                        # Update category
                        intent.category = category
                        intent.save()
                    except Intent.DoesNotExist:
                        messages.error(request, f'Entry {index}: Selected intent not found.')
                        return redirect('add_training_data')
                else:
                    messages.error(request, f'Entry {index}: Please select an intent or create a new one.')
                    return redirect('add_training_data')
                
                # Check if training data exists for this intent
                existing = TrainingData.objects.filter(intent=intent, is_active=True).first()
                
                if existing:
                    # Replace existing questions
                    existing.set_questions(questions_list)
                    existing.answer = answer
                    existing.is_active = True
                    existing.save()
                    
                    created_entries.append(f"{intent.name} (updated with {len(questions_list)} questions)")
                    entry_count += 1
                else:
                    # Create new training data
                    training_data = TrainingData.objects.create(
                        intent=intent,
                        answer=answer,
                        is_active=True
                    )
                    training_data.set_questions(questions_list)
                    training_data.save()
                    
                    created_entries.append(f"{intent.name} ({len(questions_list)} questions)")
                    entry_count += 1
                
                index += 1
            
            if entry_count > 0:
                messages.success(
                    request,
                    f'✓ Successfully added/updated {entry_count} training data entries! '
                    f'Entries: {", ".join(created_entries)}. '
                    f'Remember to retrain the model to activate changes.'
                )
            else:
                messages.warning(request, 'No entries were added.')
            
            return redirect('admin_dashboard')
            
        except Exception as e:
            messages.error(request, f'Error adding training data: {str(e)}')
            import traceback
            print(traceback.format_exc())
            return redirect('add_training_data')
    
    # GET request
    intents = Intent.objects.all().select_related('category').order_by('name')
    categories = Category.objects.filter(is_active=True).order_by('order', 'name')
    
    # Get training data for each intent
    intents_with_data = []
    for intent in intents:
        training = TrainingData.objects.filter(intent=intent, is_active=True).first()
        intent_data = {
            'id': intent.id,
            'name': intent.name,
            'description': intent.description,
            'category_id': intent.category.id if intent.category else None,
            'has_training': training is not None,
        }
        
        if training:
            intent_data['questions'] = training.get_questions()
            intent_data['answer'] = training.answer
        
        intents_with_data.append(intent_data)
    
    categories_data = [
        {
            'id': cat.id,
            'name': cat.name,
            'description': cat.description
        }
        for cat in categories
    ]
    
    intents_json = json.dumps(intents_with_data)
    categories_json = json.dumps(categories_data)
    
    return render(request, 'add_data.html', {
        'intents_json': intents_json,
        'categories_json': categories_json,
    })
    
@login_required
def create_intent_ajax(request):
    """AJAX endpoint to create new intent"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name', '').strip()
            description = data.get('description', '').strip()
            
            # Validation
            if not name:
                return JsonResponse({'success': False, 'error': 'Intent name is required'})
            
            # Check if intent already exists
            if Intent.objects.filter(name=name).exists():
                return JsonResponse({'success': False, 'error': f'Intent "{name}" already exists'})
            
            # Create intent
            intent = Intent.objects.create(
                name=name,
                description=description
            )
            
            return JsonResponse({
                'success': True,
                'intent': {
                    'id': intent.id,
                    'name': intent.name,
                    'description': intent.description
                }
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def train_model(request):
    """Trigger model training"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')
    
    try:
        # Use hybrid pipeline
        pipeline = HybridChatbotPipeline()
        accuracy, version = pipeline.train()
        
        # Reload predictor with new model
        predictor = HybridChatbotPredictor()
        predictor.reload_model()
        
        messages.success(
            request,
            f'✓ Hybrid model trained successfully! Version: {version}, Accuracy: {accuracy*100:.2f}%'
        )
    except Exception as e:
        messages.error(request, f'Training failed: {str(e)}')
    
    return redirect('admin_dashboard')

# Update chat_api view
@csrf_exempt
def chat_api(request):
    """
    Enhanced API endpoint for two-stage chat:
    1. Initial query -> Returns category for user to click
    2. Category click -> Returns full answer
    3. Location click -> Returns location details
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')
        request_type = data.get('request_type', 'initial')
        intent_name = data.get('intent', None)
        room_number = data.get('room_number', None)  # NEW: For location requests
        
        if not user_message and request_type == 'initial':
            return JsonResponse({'error': 'No message provided'}, status=400)
        
        # Get prediction using hybrid predictor
        predictor = HybridChatbotPredictor()
        
        # =====================================================
        # NEW: Handle location-specific requests
        # =====================================================
        if request_type == 'get_location' and room_number:
            result = predictor.get_location_answer(room_number)
            
            # Log the conversation
            ChatLog.objects.create(
                user_message=f"[Location: {room_number}]",
                bot_response=result['response'],
                predicted_intent='ask_room_location',
                confidence=result.get('confidence'),
            )
            
            return JsonResponse({
                'response': result['response'],
                'response_type': result.get('response_type', 'answer'),
                'confidence': round(result.get('confidence', 0), 2)
            })
        
        # =====================================================
        # Handle regular intent answer requests
        # =====================================================
        elif request_type == 'get_answer' and intent_name:
            # User clicked on category button - return the answer
            result = predictor.get_answer_for_intent(intent_name)
            
            # Log the conversation
            ChatLog.objects.create(
                user_message=f"[Clicked: {intent_name}]",
                bot_response=result['response'],
                predicted_intent=result.get('intent'),
                confidence=result.get('confidence'),
            )
            
            return JsonResponse({
                'response': result['response'],
                'intent': result.get('intent'),
                'confidence': round(result.get('confidence', 0), 2),
                'response_type': result.get('response_type', 'answer')
            })
        
        # =====================================================
        # Handle initial query - classify and return category
        # =====================================================
        else:
            result = predictor.predict(user_message, request_type='initial')
            
            # Log the conversation
            ChatLog.objects.create(
                user_message=user_message,
                bot_response=result.get('response') or f"Category: {result.get('category_display', 'Unknown')}",
                predicted_intent=result.get('intent'),
                confidence=result.get('confidence'),
            )
            
            response_data = {
                'confidence': round(result.get('confidence', 0), 2),
                'response_type': result.get('response_type', 'unknown')
            }
            
            # Add different fields based on response type
            if result.get('response_type') == 'category':
                # Return category info for display
                response_data.update({
                    'intent': result.get('intent'),
                    'intent_display': result.get('intent_display'),
                    'category': result.get('category'),
                    'category_display': result.get('category_display'),
                })
            elif result.get('response_type') == 'location_category':
                # NEW: Location category response
                response_data.update({
                    'intent': result.get('intent'),
                    'category': result.get('category'),
                    'category_display': result.get('category_display'),
                    'matched_location': result.get('matched_location'),
                })
            else:
                # Direct response (greeting, goodbye, error)
                response_data['response'] = result.get('response')
                response_data['intent'] = result.get('intent')
            
            return JsonResponse(response_data)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def chat_interface(request):
    """Chat interface for users"""
    return render(request, 'chat_interface.html')

def manage_responses(request, intent_id):
    """Manage responses for a specific intent"""
    intent = Intent.objects.get(id=intent_id)
    
    if request.method == 'POST':
        answer = request.POST.get('answer')
        is_default = request.POST.get('is_default') == 'on'
        priority = request.POST.get('priority', 0)
        
        # If this is marked as default, unmark others
        if is_default:
            IntentResponse.objects.filter(intent=intent).update(is_default=False)
        
        IntentResponse.objects.create(
            intent=intent,
            answer=answer,
            is_default=is_default,
            priority=priority
        )
        messages.success(request, 'Response added successfully!')
        return redirect('manage_responses', intent_id=intent_id)
    
    responses = IntentResponse.objects.filter(intent=intent)
    return render(request, 'manage_responses.html', {
        'intent': intent,
        'responses': responses
    })


@login_required
def office_dashboard(request):
    """Dashboard for office staff"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        office = office_user.office
    except OfficeUser.DoesNotExist:
        messages.error(request, 'You are not associated with any office.')
        return redirect('chat_interface')
    
    # Get training data with pagination
    training_data_list = TrainingData.objects.filter(
        office=office,
        submitted_by=request.user
    ).select_related('intent', 'reviewed_by').order_by('-submitted_at')
    
    training_paginator = Paginator(training_data_list, 5)  # 5 per page
    training_page = request.GET.get('training_page')
    
    try:
        training_data = training_paginator.page(training_page)
    except PageNotAnInteger:
        training_data = training_paginator.page(1)
    except EmptyPage:
        training_data = training_paginator.page(training_paginator.num_pages)
    
    # Get locations with pagination
    locations_list = Location.objects.filter(
        is_active=True
    ).prefetch_related('keywords').order_by('-created_at')
    
    location_paginator = Paginator(locations_list, 5)  # 5 per page
    location_page = request.GET.get('location_page')
    
    try:
        locations = location_paginator.page(location_page)
    except PageNotAnInteger:
        locations = location_paginator.page(1)
    except EmptyPage:
        locations = location_paginator.page(location_paginator.num_pages)
    
    # Statistics
    training_stats = {
        'total': training_data_list.count(),
        'pending': training_data_list.filter(is_reviewed=False).count(),
        'reviewed': training_data_list.filter(is_reviewed=True).count(),
    }
    
    location_stats = {
        'total': locations_list.count(),
    }
    
    return render(request, 'office_dashboard.html', {
        'office': office,
        'training_data': training_data,
        'locations': locations,
        'training_stats': training_stats,
        'location_stats': location_stats,
    })


@login_required
def ticket_detail(request, ticket_id):
    """View ticket details and add comments"""
    ticket = get_object_or_404(DataTicket, id=ticket_id)
    
    # Check permissions
    try:
        office_user = request.user.office_profile
        if ticket.office != office_user.office and not request.user.is_staff:
            messages.error(request, 'You do not have permission to view this ticket.')
            return redirect('office_dashboard')
    except OfficeUser.DoesNotExist:
        if not request.user.is_staff:
            messages.error(request, 'Access denied.')
            return redirect('chat_interface')
    
    if request.method == 'POST':
        comment_text = request.POST.get('comment')
        if comment_text:
            TicketComment.objects.create(
                ticket=ticket,
                user=request.user,
                comment=comment_text
            )
            messages.success(request, 'Comment added.')
            return redirect('ticket_detail', ticket_id=ticket_id)
    
    comments = ticket.comments.all()
    context = {
        'ticket': ticket,
        'comments': comments,
    }
    return render(request, 'ticket_detail.html', context)

@login_required
def admin_ticket_review(request):

    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('office_dashboard')


    status_filter = request.GET.get('status', 'pending')
    ticket_type = request.GET.get('type', 'data')


    # -----------------------------
    # LOCATION TICKETS
    # -----------------------------
    if ticket_type == 'location':

        tickets = LocationTicket.objects.filter(
            status=status_filter
        ).select_related(
            'office',
            'reviewed_by'
        ).order_by('-priority', '-created_at')


        stats = {
            'pending': LocationTicket.objects.filter(status='pending').count(),
            'approved': LocationTicket.objects.filter(status='approved').count(),
            'rejected': LocationTicket.objects.filter(status='rejected').count(),
            'implemented': LocationTicket.objects.filter(status='implemented').count(),
        }


    # -----------------------------
    # UPDATE TICKETS (NEW)
    # -----------------------------
    elif ticket_type == 'update':

        tickets = TrainingUpdateTicket.objects.filter(
            status=status_filter
        ).select_related(
            'office',
            'training_data__intent',
            'reviewed_by'
        ).order_by('-created_at')


        stats = {
            'pending': TrainingUpdateTicket.objects.filter(status='pending').count(),
            'approved': TrainingUpdateTicket.objects.filter(status='approved').count(),
            'rejected': TrainingUpdateTicket.objects.filter(status='rejected').count(),
        }


    # -----------------------------
    # DATA TICKETS (DEFAULT)
    # -----------------------------
    else:

        tickets = DataTicket.objects.filter(
            status=status_filter
        ).select_related(
            'office',
            'intent'
        ).order_by('-priority', '-created_at')


        stats = {
            'pending': DataTicket.objects.filter(status='pending').count(),
            'approved': DataTicket.objects.filter(status='approved').count(),
            'rejected': DataTicket.objects.filter(status='rejected').count(),
            'implemented': DataTicket.objects.filter(status='implemented').count(),
        }


    return render(request, 'admin_ticket_review.html', {
        'tickets': tickets,
        'stats': stats,
        'current_status': status_filter,
        'ticket_type': ticket_type,
    })



@login_required
def approve_ticket(request, ticket_id):
    """Admin approves ticket and adds to training data"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('office_dashboard')
    
    ticket = get_object_or_404(DataTicket, id=ticket_id)
    
    if request.method == 'POST':
        admin_notes = request.POST.get('admin_notes', '')
        
        # Create or get intent
        if ticket.new_intent_name:
            intent, created = Intent.objects.get_or_create(
                name=ticket.new_intent_name,
                defaults={'description': ticket.new_intent_description}
            )
            ticket.intent = intent
        
        if not ticket.intent:
            messages.error(request, 'Intent is required to approve ticket.')
            return redirect('admin_ticket_review')
        
        # Split questions by separator (from ticket submission)
        questions = ticket.question.split('\n---\n')
        questions = [q.strip() for q in questions if q.strip()]
        
        # Check if training data for this intent already exists
        existing = TrainingData.objects.filter(
            intent=ticket.intent,
            is_active=True
        ).first()
        
        if existing:
            # Intent already has training data - add questions to it
            current_questions = existing.get_questions()
            new_questions = [q for q in questions if q not in current_questions]
            
            if new_questions:
                current_questions.extend(new_questions)
                existing.set_questions(current_questions)
                existing.answer = ticket.answer  # Update answer
                existing.save()
                
                training_data = existing
                created_count = len(new_questions)
                
                messages.success(
                    request,
                    f'Added {created_count} new question(s) to existing {ticket.intent.name} training data. '
                    f'Total: {len(current_questions)} questions.'
                )
            else:
                training_data = existing
                created_count = 0
                messages.info(request, 'All questions already exist for this intent.')
        else:
            # Create new training data entry
            training_data = TrainingData.objects.create(
                intent=ticket.intent,
                answer=ticket.answer,
                is_active=True
            )
            training_data.set_questions(questions)
            training_data.save()
            created_count = len(questions)
            
            messages.success(
                request,
                f'Created new training data for {ticket.intent.name} with {created_count} question(s)!'
            )
        
        # Update ticket
        ticket.status = 'approved'
        ticket.admin_notes = admin_notes
        ticket.reviewed_by = request.user
        ticket.reviewed_at = timezone.now()
        ticket.training_data = training_data
        ticket.save()
        
        # Add comment
        TicketComment.objects.create(
            ticket=ticket,
            user=request.user,
            comment=f'Ticket approved. Added {created_count} training sample(s). {admin_notes}'
        )
        
        return redirect('admin_ticket_review')
    
    # Split questions for display
    questions = ticket.question.split('\n---\n')
    questions = [q.strip() for q in questions if q.strip()]
    
    context = {
        'ticket': ticket,
        'questions': questions,
    }
    return render(request, 'approve_ticket.html', context)

@login_required
def reject_ticket(request, ticket_id):
    """Admin rejects ticket"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('office_dashboard')
    
    ticket = get_object_or_404(DataTicket, id=ticket_id)
    
    if request.method == 'POST':
        admin_notes = request.POST.get('admin_notes', '')
        
        ticket.status = 'rejected'
        ticket.admin_notes = admin_notes
        ticket.reviewed_by = request.user
        ticket.reviewed_at = timezone.now()
        ticket.save()
        
        # Add comment
        TicketComment.objects.create(
            ticket=ticket,
            user=request.user,
            comment=f'Ticket rejected. Reason: {admin_notes}'
        )
        
        messages.success(request, f'Ticket {ticket.ticket_number} rejected.')
        return redirect('admin_ticket_review')
    
    context = {
        'ticket': ticket,
    }
    return render(request, 'reject_ticket.html', context)

@login_required
def bulk_approve_tickets(request):

    if not request.user.is_staff:
        return JsonResponse({'error': 'Admin access required'}, status=403)

    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    try:
        data = json.loads(request.body)
        ticket_ids = data.get('ticket_ids', [])

        approved_count = 0
        errors = []

        for ticket_id in ticket_ids:

            try:
                ticket = DataTicket.objects.get(
                    id=ticket_id,
                    status='pending'
                )

                # Create / get intent
                if ticket.new_intent_name:
                    intent, _ = Intent.objects.get_or_create(
                        name=ticket.new_intent_name,
                        defaults={
                            'description': ticket.new_intent_description
                        }
                    )
                    ticket.intent = intent

                if not ticket.intent:
                    errors.append(f'Ticket {ticket.id}: No intent')
                    continue


                # Split questions (same as approve_ticket)
                questions = ticket.question.split('\n---\n')
                questions = [q.strip() for q in questions if q.strip()]


                # Check existing training data
                existing = TrainingData.objects.filter(
                    intent=ticket.intent,
                    is_active=True
                ).first()


                if existing:
                    # Add to existing

                    current_questions = existing.get_questions()

                    new_questions = [
                        q for q in questions
                        if q not in current_questions
                    ]

                    if new_questions:
                        current_questions.extend(new_questions)

                        existing.set_questions(current_questions)
                        existing.answer = ticket.answer
                        existing.save()

                        training_data = existing

                    else:
                        training_data = existing


                else:
                    # Create new training data

                    training_data = TrainingData.objects.create(
                        intent=ticket.intent,
                        answer=ticket.answer,
                        is_active=True
                    )

                    training_data.set_questions(questions)
                    training_data.save()   # ✅ IMPORTANT


                # Update ticket
                ticket.status = 'approved'
                ticket.reviewed_by = request.user
                ticket.reviewed_at = timezone.now()
                ticket.training_data = training_data
                ticket.save()

                approved_count += 1


            except DataTicket.DoesNotExist:
                errors.append(f'Ticket {ticket_id} not found')
                continue

            except Exception as e:
                errors.append(str(e))
                continue


        return JsonResponse({
            'success': True,
            'approved_count': approved_count,
            'errors': errors,
            'message': f'{approved_count} ticket(s) approved successfully!'
        })


    except Exception as e:

        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def user_login(request):
    """User login view"""
    if request.user.is_authenticated:
        # Redirect based on user type
        try:
            office_user = request.user.office_profile
            return redirect('office_dashboard')
        except:
            if request.user.is_staff:
                return redirect('admin_dashboard')
            return redirect('chat_interface')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            next_url = request.POST.get('next') or request.GET.get('next')
            
            if next_url:
                return redirect(next_url)
            
            # Redirect based on user type
            try:
                office_user = user.office_profile
                return redirect('office_dashboard')
            except:
                if user.is_staff:
                    return redirect('admin_dashboard')
                return redirect('chat_interface')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'login.html')


def user_logout(request):
    """User logout view"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')


@login_required
def manage_locations(request):
    """Manage room/location data (Admin)"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')
    
    if request.method == 'POST':
        # Handle batch location addition
        try:
            entry_count = 0
            created_entries = []
            skipped_entries = []
            index = 1
            
            # Loop through entries
            while f'room_number_{index}' in request.POST:
                room_number = request.POST.get(f'room_number_{index}', '').strip()
                room_name = request.POST.get(f'room_name_{index}', '').strip()
                building = request.POST.get(f'building_{index}', '').strip()
                floor = request.POST.get(f'floor_{index}', '').strip()
                description = request.POST.get(f'description_{index}', '').strip()
                
                # Get keywords array
                keywords_list = request.POST.getlist(f'keywords_{index}[]')
                keywords_list = [k.strip().lower() for k in keywords_list if k.strip()]
                
                # Validation
                if not room_number or not building or not floor:
                    messages.error(request, f'Location {index}: Room number, building, and floor are required.')
                    return redirect('manage_locations')
                
                if not keywords_list:
                    messages.error(request, f'Location {index}: Please provide at least one keyword.')
                    return redirect('manage_locations')
                
                # Check if location already exists
                if Location.objects.filter(room_number=room_number).exists():
                    skipped_entries.append(f"{room_number} (already exists)")
                    index += 1
                    continue
                
                # Create location
                location = Location.objects.create(
                    room_number=room_number,
                    room_name=room_name,
                    building=building,
                    floor=floor,
                    description=description,
                    is_active=True
                )
                
                # Always add room number as first keyword
                if room_number.lower() not in keywords_list:
                    keywords_list.insert(0, room_number.lower())
                
                # Create keywords
                for keyword in keywords_list:
                    LocationKeyword.objects.create(
                        location=location,
                        keyword=keyword,
                        priority=1
                    )
                
                created_entries.append(f"{room_number} ({len(keywords_list)} keywords)")
                entry_count += 1
                
                index += 1
            
            # Reload location extractor
            if entry_count > 0:
                try:
                    from .hybrid_predictor import HybridChatbotPredictor
                    predictor = HybridChatbotPredictor()
                    predictor.location_extractor._load_locations()
                except Exception as e:
                    print(f"Warning: Could not reload location extractor: {e}")
            
            # Success message
            if entry_count > 0:
                success_msg = f'✓ Successfully added {entry_count} location(s)! Entries: {", ".join(created_entries)}. '
                if skipped_entries:
                    success_msg += f'Skipped {len(skipped_entries)} duplicate(s): {", ".join(skipped_entries)}. '
                messages.success(request, success_msg)
            elif skipped_entries:
                messages.warning(request, f'All {len(skipped_entries)} location(s) already exist: {", ".join(skipped_entries)}')
            else:
                messages.warning(request, 'No locations were added.')
            
            return redirect('manage_locations')
            
        except Exception as e:
            messages.error(request, f'Error adding locations: {str(e)}')
            return redirect('manage_locations')
    
    # GET request
    locations = Location.objects.all().prefetch_related('keywords').order_by('building', 'floor', 'room_number')
    
    return render(request, 'manage_locations.html', {
        'locations': locations
    })


@login_required
def admin_location_detail(request, location_id):
    """Get location details for editing (AJAX) - Admin"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Not authorized'}, status=403)
    
    try:
        location = Location.objects.get(id=location_id)
        keywords = [kw.keyword for kw in location.keywords.all()]
        
        return JsonResponse({
            'success': True,
            'location': {
                'id': location.id,
                'room_number': location.room_number,
                'room_name': location.room_name or '',
                'building': location.building,
                'floor': location.floor,
                'description': location.description or '',
                'keywords': keywords
            }
        })
    except Location.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Location not found'}, status=404)


@login_required
@require_http_methods(["POST"])
def update_location(request, location_id):
    """Admin update location"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    try:
        location = Location.objects.get(id=location_id)
        data = json.loads(request.body)

        room_number = data.get('room_number', '').strip()
        room_name = data.get('room_name', '').strip()
        building = data.get('building', '').strip()
        floor = data.get('floor', '').strip()
        description = data.get('description', '').strip()
        keywords = data.get('keywords', [])

        if not room_number or not building or not floor or not keywords:
            return JsonResponse({'success': False, 'error': 'Missing required fields'})

        # Check duplicate (exclude current location)
        if Location.objects.exclude(id=location.id).filter(room_number=room_number).exists():
            return JsonResponse({'success': False, 'error': 'Room number already exists'})

        # Update location
        location.room_number = room_number
        location.room_name = room_name
        location.building = building
        location.floor = floor
        location.description = description
        location.save()

        # Update keywords
        location.keywords.all().delete()
        
        for keyword in keywords:
            if keyword.strip():
                LocationKeyword.objects.create(
                    location=location,
                    keyword=keyword.strip().lower(),
                    priority=1
                )

        # Reload location extractor
        try:
            from .hybrid_predictor import HybridChatbotPredictor
            predictor = HybridChatbotPredictor()
            predictor.location_extractor._load_locations()
        except Exception as e:
            print(f"Warning: Could not reload location extractor: {e}")

        return JsonResponse({'success': True})

    except Location.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Location not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
def delete_location(request, location_id):
    """Admin delete location"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        location = Location.objects.get(id=location_id)
        room_number = location.room_number
        location.delete()
        
        # Reload location extractor
        try:
            from .hybrid_predictor import HybridChatbotPredictor
            predictor = HybridChatbotPredictor()
            predictor.location_extractor._load_locations()
        except Exception as e:
            print(f"Warning: Could not reload location extractor: {e}")
        
        return JsonResponse({
            'success': True,
            'message': f'Location {room_number} deleted successfully'
        })
    except Location.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Location not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
@csrf_exempt
def submit_feedback(request):
    """Handle feedback submission from users"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            feedback_type = data.get('type')
            rating = data.get('rating')
            message = data.get('message', '')
            
            # Get IP address
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                ip_address = x_forwarded_for.split(',')[0]
            else:
                ip_address = request.META.get('REMOTE_ADDR')
            
            # Create feedback
            feedback = Feedback.objects.create(
                feedback_type=feedback_type,
                rating=rating if feedback_type == 'rating' else None,
                message=message,
                user=request.user if request.user.is_authenticated else None,
                ip_address=ip_address
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Feedback submitted successfully',
                'feedback_id': feedback.id
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)


@login_required
def manage_feedback(request):
    """Admin page to view and manage all feedback"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')
    
    # Get all feedback
    feedbacks_list = Feedback.objects.all().select_related('user')
    
    # Statistics
    total_ratings = Feedback.objects.filter(feedback_type='rating').count()
    total_suggestions = Feedback.objects.filter(feedback_type='suggestion').count()
    total_reports = Feedback.objects.filter(feedback_type='report').count()
    unresolved_count = Feedback.objects.filter(is_resolved=False).count()
    
    # Average rating
    avg_rating = Feedback.objects.filter(
        feedback_type='rating',
        rating__isnull=False
    ).aggregate(Avg('rating'))['rating__avg'] or 0
    
    # Pagination
    paginator = Paginator(feedbacks_list, 20)
    page_number = request.GET.get('page')
    feedbacks = paginator.get_page(page_number)
    
    return render(request, 'manage_feedback.html', {
        'feedbacks': feedbacks,
        'total_ratings': total_ratings,
        'total_suggestions': total_suggestions,
        'total_reports': total_reports,
        'unresolved_count': unresolved_count,
        'average_rating': avg_rating,
    })


@login_required
@require_http_methods(["POST"])
def resolve_feedback(request, feedback_id):
    """Mark feedback as resolved"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.is_resolved = True
        feedback.save()
        
        return JsonResponse({'success': True})
    except Feedback.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Feedback not found'})


@login_required
def feedback_details(request, feedback_id):
    """Get feedback details"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        feedback = Feedback.objects.get(id=feedback_id)
        
        return JsonResponse({
            'success': True,
            'feedback_type': feedback.get_feedback_type_display(),
            'rating': feedback.rating,
            'message': feedback.message,
            'created_at': feedback.created_at.strftime('%B %d, %Y at %I:%M %p'),
            'user': feedback.user.username if feedback.user else None,
            'ip_address': feedback.ip_address,
            'admin_notes': feedback.admin_notes,
            'is_resolved': feedback.is_resolved,
        })
    except Feedback.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Feedback not found'})


@login_required
@require_http_methods(["POST"])
def save_feedback_notes(request, feedback_id):
    """Save admin notes for feedback"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        data = json.loads(request.body)
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.admin_notes = data.get('notes', '')
        feedback.save()
        
        return JsonResponse({'success': True})
    except Feedback.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Feedback not found'})


@login_required
@require_http_methods(["DELETE"])
def delete_feedback(request, feedback_id):
    """Delete feedback"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        feedback = Feedback.objects.get(id=feedback_id)
        feedback.delete()
        
        return JsonResponse({'success': True})
    except Feedback.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Feedback not found'})
    
    
@login_required
def manage_users(request):

    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')

    # ---- SEARCH PARAMETERS ----
    user_query = request.GET.get('user_search', '')
    office_query = request.GET.get('office_search', '')

    users = User.objects.select_related(
        'office_profile__office'
    ).order_by('-date_joined')

    offices = Office.objects.all().order_by('name')

    # ---- FILTER USERS ----
    if user_query:
        users = users.filter(
            Q(username__icontains=user_query) |
            Q(first_name__icontains=user_query) |
            Q(last_name__icontains=user_query) |
            Q(email__icontains=user_query)
        )

    # ---- FILTER OFFICES ----
    if office_query:
        offices = offices.filter(
            Q(name__icontains=office_query) |
            Q(description__icontains=office_query) |
            Q(contact_email__icontains=office_query)
        )

    # ---- PAGINATION ----
    user_paginator = Paginator(users, 5)
    office_paginator = Paginator(offices, 5)

    user_page_number = request.GET.get('user_page')
    office_page_number = request.GET.get('office_page')

    users_page = user_paginator.get_page(user_page_number)
    offices_page = office_paginator.get_page(office_page_number)

    return render(request, 'manage_users.html', {
        'users': users_page,
        'offices': offices_page,
        'user_query': user_query,
        'office_query': office_query
    })
    
@login_required
@require_http_methods(["POST"])
def create_office(request):
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})

    try:
        data = json.loads(request.body)

        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        email = data.get('contact_email', '').strip()

        if not name:
            return JsonResponse({'success': False, 'error': 'Office name is required'})

        if Office.objects.filter(name__iexact=name).exists():
            return JsonResponse({'success': False, 'error': 'Office already exists'})

        Office.objects.create(
            name=name,
            description=description,
            contact_email=email
        )

        return JsonResponse({'success': True})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
@login_required
@require_http_methods(["POST"])
def update_office(request, office_id):

    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})

    try:
        office = Office.objects.get(id=office_id)
        data = json.loads(request.body)

        office.name = data.get('name')
        office.description = data.get('description', '')
        office.contact_email = data.get('contact_email', '')
        office.save()

        return JsonResponse({'success': True})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
@login_required
@require_http_methods(["DELETE"])
def delete_office(request, office_id):

    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})

    try:
        office = Office.objects.get(id=office_id)

        if office.staff.exists():
            return JsonResponse({
                'success': False,
                'error': 'Cannot delete office with assigned staff.'
            })

        office.delete()

        return JsonResponse({'success': True})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
def register_user(request):
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')

    offices = Office.objects.all()

    if request.method == 'POST':

        form_data = request.POST.dict()  # keep user input

        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        password = request.POST.get('password')
        user_type = request.POST.get('user_type')
        office_id = request.POST.get('office')
        new_office_name = request.POST.get('new_office_name', '').strip()
        new_office_desc = request.POST.get('new_office_desc', '').strip()
        new_office_email = request.POST.get('new_office_email', '').strip()

        # Validation
        if not all([username, first_name, last_name, password, user_type]):
            messages.error(request, 'All required fields must be filled.')
            return render(request, 'register_user.html', {
                'offices': offices,
                'form_data': form_data
            })

        if len(password) < 8:
            messages.error(request, 'Password must be at least 8 characters.')
            return render(request, 'register_user.html', {
                'offices': offices,
                'form_data': form_data
            })

        if User.objects.filter(username=username).exists():
            messages.error(request, f'Username "{username}" already exists.')
            return render(request, 'register_user.html', {
                'offices': offices,
                'form_data': form_data
            })

        if user_type == 'staff' and not office_id and not new_office_name:
            messages.error(request, 'Office staff must be assigned to an office or create a new one.')
            return render(request, 'register_user.html', {
                'offices': offices,
                'form_data': form_data
            })

        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )

        if user_type == 'admin':
            user.is_staff = True
            user.save()

        elif user_type == 'staff':

            if office_id:
                office = Office.objects.get(id=office_id)
            else:
                if Office.objects.filter(name__iexact=new_office_name).exists():
                    messages.error(request, 'Office name already exists.')
                    user.delete()
                    return render(request, 'register_user.html', {
                        'offices': offices,
                        'form_data': form_data
                    })

                office = Office.objects.create(
                    name=new_office_name,
                    description=new_office_desc,
                    contact_email=new_office_email
                )

            OfficeUser.objects.create(user=user, office=office)

        messages.success(request, f'✓ User "{username}" created successfully!')
        return redirect('manage_users')

    return render(request, 'register_user.html', {
        'offices': offices
    })


@login_required
@require_http_methods(["POST"])
def update_user_ajax(request, user_id):

    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'})

    try:
        user = get_object_or_404(User, id=user_id)

        data = json.loads(request.body)

        user.first_name = data.get('first_name', '').strip()
        user.last_name = data.get('last_name', '').strip()
        user.email = data.get('email', '').strip()

        password = data.get('password', '').strip()

        if password:
            if len(password) < 8:
                return JsonResponse({
                    'success': False,
                    'error': 'Password must be at least 8 characters'
                })

            user.set_password(password)


        role = data.get('role')

        if role == 'admin':
            user.is_staff = True

        else:
            user.is_staff = False


        user.save()


        # Office handling
        if role == 'staff':

            office_id = data.get('office')

            if not office_id:
                return JsonResponse({
                    'success': False,
                    'error': 'Staff must have office'
                })

            office = Office.objects.get(id=office_id)

            OfficeUser.objects.update_or_create(
                user=user,
                defaults={'office': office}
            )

        else:
            OfficeUser.objects.filter(user=user).delete()


        return JsonResponse({'success': True})


    except Exception as e:

        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@login_required
@require_http_methods(["DELETE"])
def delete_user(request, user_id):
    """Delete a user"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        user = User.objects.get(id=user_id)
        
        # Prevent deleting yourself
        if user == request.user:
            return JsonResponse({'success': False, 'error': 'You cannot delete your own account'})
        
        # Prevent deleting superusers
        if user.is_superuser:
            return JsonResponse({'success': False, 'error': 'Cannot delete superuser accounts'})
        
        username = user.username
        user.delete()
        
        return JsonResponse({'success': True, 'message': f'User {username} deleted'})
    except User.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'User not found'})
    
@login_required
def submit_location_ticket(request):
    """Office staff submit location tickets"""
    # Get user's office
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        office = office_user.office
    except OfficeUser.DoesNotExist:
        messages.error(request, 'You are not associated with any office.')
        return redirect('chat_interface')
    
    if request.method == 'POST':
        try:
            # Count how many entries were submitted
            entry_count = 0
            index = 1
            
            while f'room_number_{index}' in request.POST:
                entry_count += 1
                
                room_number = request.POST.get(f'room_number_{index}', '').strip()
                room_name = request.POST.get(f'room_name_{index}', '').strip()
                building = request.POST.get(f'building_{index}', '').strip()
                floor = request.POST.get(f'floor_{index}', '').strip()
                room_description = request.POST.get(f'room_description_{index}', '').strip()
                keywords = request.POST.get(f'keywords_{index}', '').strip()
                priority = request.POST.get(f'priority_{index}', 'medium')
                notes = request.POST.get(f'notes_{index}', '').strip()
                
                # Validation
                if not all([room_number, building, floor, keywords]):
                    messages.error(request, f'Entry {index}: Room number, building, floor, and keywords are required.')
                    return redirect('submit_location_ticket')
                
                # Create location ticket
                LocationTicket.objects.create(
                    submitted_by=request.user,
                    office=office,
                    room_number=room_number,
                    room_name=room_name,
                    building=building,
                    floor=floor,
                    room_description=room_description,
                    keywords=keywords,
                    priority=priority,
                    notes=notes,
                    status='pending'
                )
                
                index += 1
            
            if entry_count > 0:
                messages.success(request, f'✓ Successfully submitted {entry_count} location ticket(s) for review!')
            else:
                messages.error(request, 'No entries were submitted.')
            
            return redirect('office_dashboard')
            
        except Exception as e:
            messages.error(request, f'Error submitting tickets: {str(e)}')
            return redirect('submit_location_ticket')
    
    # GET request
    return render(request, 'submit_location_ticket.html')


@login_required
def location_ticket_detail(request, ticket_id):
    """View details of a specific location ticket"""
    ticket = get_object_or_404(LocationTicket, id=ticket_id)
    
    # Check if user has access to this ticket
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        if ticket.office != office_user.office and not request.user.is_staff:
            messages.error(request, 'You do not have permission to view this ticket.')
            return redirect('office_dashboard')
    except OfficeUser.DoesNotExist:
        if not request.user.is_staff:
            messages.error(request, 'Access denied.')
            return redirect('chat_interface')
    
    # Get comments
    comments = LocationTicketComment.objects.filter(ticket=ticket).select_related('user')
    
    # Pre-process keywords for template
    keywords_list = [k.strip() for k in ticket.keywords.split(',')] if ticket.keywords else []
    
    return render(request, 'location_ticket_detail.html', {
        'ticket': ticket,
        'comments': comments,
        'keywords_list': keywords_list,
    })
    
@login_required
def approve_location_ticket(request, ticket_id):
    """Admin approves location ticket and creates location"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('office_dashboard')
    
    ticket = get_object_or_404(LocationTicket, id=ticket_id)
    
    if request.method == 'POST':
        admin_notes = request.POST.get('admin_notes', '')
        
        try:
            # Check if location already exists
            if Location.objects.filter(room_number=ticket.room_number).exists():
                messages.error(request, f'Location with room number "{ticket.room_number}" already exists.')
                return redirect('approve_location_ticket', ticket_id=ticket.id)
            
            # Create location
            location = Location.objects.create(
                room_number=ticket.room_number,
                room_name=ticket.room_name,
                building=ticket.building,
                floor=ticket.floor,
                description=ticket.room_description,
                is_active=True
            )
            
            # Create keywords
            keywords_list = [k.strip().lower() for k in ticket.keywords.split(',') if k.strip()]
            
            # Always add room number as keyword
            if ticket.room_number.lower() not in keywords_list:
                keywords_list.insert(0, ticket.room_number.lower())
            
            keyword_count = 0
            for keyword in keywords_list:
                LocationKeyword.objects.create(
                    location=location,
                    keyword=keyword,
                    priority=1
                )
                keyword_count += 1
            
            # Update ticket
            ticket.status = 'approved'
            ticket.admin_notes = admin_notes
            ticket.reviewed_by = request.user
            ticket.reviewed_at = timezone.now()
            ticket.location = location
            ticket.save()
            
            # Add comment
            LocationTicketComment.objects.create(
                ticket=ticket,
                user=request.user,
                comment=f'Ticket approved. Created location with {keyword_count} keywords. {admin_notes}'
            )
            
            # Reload location extractor
            from .hybrid_predictor import HybridChatbotPredictor
            predictor = HybridChatbotPredictor()
            predictor.location_extractor._load_locations()
            
            messages.success(
                request,
                f'Location "{ticket.room_number}" created successfully with {keyword_count} keywords! '
                f'It is now available in the chatbot.'
            )
            
            return redirect('admin_ticket_review')
            
        except Exception as e:
            messages.error(request, f'Error creating location: {str(e)}')
            return redirect('approve_location_ticket', ticket_id=ticket.id)
    
    # GET request - show approval form
    # Pre-process keywords for display
    keywords_list = [k.strip() for k in ticket.keywords.split(',')] if ticket.keywords else []
    
    context = {
        'ticket': ticket,
        'keywords_list': keywords_list,
    }
    return render(request, 'approve_location_ticket.html', context)


@login_required
def reject_location_ticket(request, ticket_id):
    """Admin rejects location ticket"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('office_dashboard')
    
    ticket = get_object_or_404(LocationTicket, id=ticket_id)
    
    if request.method == 'POST':
        admin_notes = request.POST.get('admin_notes', '')
        
        if not admin_notes.strip():
            messages.error(request, 'Please provide a reason for rejection.')
            return redirect('reject_location_ticket', ticket_id=ticket.id)
        
        ticket.status = 'rejected'
        ticket.admin_notes = admin_notes
        ticket.reviewed_by = request.user
        ticket.reviewed_at = timezone.now()
        ticket.save()
        
        # Add comment
        LocationTicketComment.objects.create(
            ticket=ticket,
            user=request.user,
            comment=f'Ticket rejected. Reason: {admin_notes}'
        )
        
        messages.success(request, f'Location ticket {ticket.ticket_number} rejected.')
        return redirect('admin_ticket_review')
    
    # GET request - show rejection form
    # Pre-process keywords for display
    keywords_list = [k.strip() for k in ticket.keywords.split(',')] if ticket.keywords else []
    
    context = {
        'ticket': ticket,
        'keywords_list': keywords_list,
    }
    return render(request, 'reject_location_ticket.html', context)


@login_required
def bulk_approve_location_tickets(request):
    """Admin bulk approves multiple location tickets at once"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Admin access required'}, status=403)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ticket_ids = data.get('ticket_ids', [])
            
            approved_count = 0
            skipped_count = 0
            errors = []
            
            for ticket_id in ticket_ids:
                try:
                    ticket = LocationTicket.objects.get(id=ticket_id, status='pending')
                    
                    # Check if location already exists
                    if Location.objects.filter(room_number=ticket.room_number).exists():
                        skipped_count += 1
                        errors.append(f'{ticket.ticket_number}: Room {ticket.room_number} already exists')
                        continue
                    
                    # Create location
                    location = Location.objects.create(
                        room_number=ticket.room_number,
                        room_name=ticket.room_name,
                        building=ticket.building,
                        floor=ticket.floor,
                        description=ticket.room_description,
                        is_active=True
                    )
                    
                    # Create keywords
                    keywords_list = [k.strip().lower() for k in ticket.keywords.split(',') if k.strip()]
                    
                    if ticket.room_number.lower() not in keywords_list:
                        keywords_list.insert(0, ticket.room_number.lower())
                    
                    for keyword in keywords_list:
                        LocationKeyword.objects.create(
                            location=location,
                            keyword=keyword,
                            priority=1
                        )
                    
                    # Update ticket
                    ticket.status = 'implemented'
                    ticket.reviewed_by = request.user
                    ticket.reviewed_at = timezone.now()
                    ticket.location = location
                    ticket.save()
                    
                    approved_count += 1
                    
                except LocationTicket.DoesNotExist:
                    skipped_count += 1
                    errors.append(f'Ticket ID {ticket_id}: Not found or already processed')
                    continue
                except Exception as e:
                    skipped_count += 1
                    errors.append(f'Ticket ID {ticket_id}: {str(e)}')
                    continue
            
            # Reload location extractor if any locations were created
            if approved_count > 0:
                from .hybrid_predictor import HybridChatbotPredictor
                predictor = HybridChatbotPredictor()
                predictor.location_extractor._load_locations()
            
            message = f'{approved_count} location(s) created successfully!'
            if skipped_count > 0:
                message += f' {skipped_count} skipped.'
            
            return JsonResponse({
                'success': True,
                'approved_count': approved_count,
                'skipped_count': skipped_count,
                'errors': errors,
                'message': message
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@login_required
def add_location_ticket_comment(request, ticket_id):
    """Add a comment to a location ticket"""
    if request.method == 'POST':
        ticket = get_object_or_404(LocationTicket, id=ticket_id)
        
        # Check if user has access
        try:
            office_user = OfficeUser.objects.get(user=request.user)
            if ticket.office != office_user.office and not request.user.is_staff:
                messages.error(request, 'Access denied.')
                return redirect('office_dashboard')
        except OfficeUser.DoesNotExist:
            if not request.user.is_staff:
                messages.error(request, 'Access denied.')
                return redirect('chat_interface')
        
        comment_text = request.POST.get('comment', '').strip()
        
        if comment_text:
            LocationTicketComment.objects.create(
                ticket=ticket,
                user=request.user,
                comment=comment_text
            )
            messages.success(request, 'Comment added successfully!')
        else:
            messages.error(request, 'Comment cannot be empty.')
        
        return redirect('location_ticket_detail', ticket_id=ticket.id)
    
    return redirect('office_dashboard')

@login_required
def submit_update_ticket(request):

    try:
        office = request.user.office_profile.office
    except:
        messages.error(request, 'No office assigned.')
        return redirect('chat_interface')


    if request.method == 'POST':

        training_id = request.POST.get('training_id')

        new_questions = request.POST.get('new_questions', '').strip()
        new_answer = request.POST.get('new_answer', '').strip()
        reason = request.POST.get('reason', '').strip()


        if not training_id or not reason:
            messages.error(request, 'Required fields missing.')
            return redirect('submit_update_ticket')


        training = get_object_or_404(
            TrainingData,
            id=training_id
        )


        TrainingUpdateTicket.objects.create(
            submitted_by=request.user,
            office=office,
            training_data=training,
            new_questions=new_questions,
            new_answer=new_answer,
            reason=reason
        )


        messages.success(request, 'Update request submitted.')

        return redirect('office_dashboard')


    training_data = TrainingData.objects.select_related('intent')


    return render(request, 'submit_update_ticket.html', {
        'training_data': training_data
    })

@login_required
def update_ticket_detail(request, ticket_id):

    if not request.user.is_staff:
        messages.error(request, 'Access denied.')
        return redirect('admin_ticket_review')


    ticket = get_object_or_404(
        TrainingUpdateTicket,
        id=ticket_id
    )


    return render(request, 'update_ticket_detail.html', {
        'ticket': ticket
    })

@login_required
def approve_update_ticket(request, ticket_id):

    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('admin_ticket_review')


    ticket = get_object_or_404(
        TrainingUpdateTicket,
        id=ticket_id,
        status='pending'
    )


    training = ticket.training_data


    # ---------------- Merge Questions ----------------

    current_questions = training.get_questions()

    new_questions = [
        q.strip()
        for q in ticket.new_questions.split('\n')
        if q.strip()
    ] if ticket.new_questions else []


    merged = list(set(current_questions + new_questions))

    training.set_questions(merged)


    # ---------------- Update Answer (if provided) ----------------

    if ticket.new_answer:
        training.answer = ticket.new_answer


    training.save()


    # ---------------- Update Ticket ----------------

    ticket.status = 'approved'
    ticket.reviewed_by = request.user
    ticket.reviewed_at = timezone.now()
    ticket.save()


    messages.success(
        request,
        'Update ticket approved and applied successfully.'
    )


    return redirect('admin_ticket_review')

@login_required
def reject_update_ticket(request, ticket_id):

    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('admin_ticket_review')


    ticket = get_object_or_404(
        TrainingUpdateTicket,
        id=ticket_id,
        status='pending'
    )


    ticket.status = 'rejected'
    ticket.reviewed_by = request.user
    ticket.reviewed_at = timezone.now()
    ticket.save()


    messages.info(request, 'Update ticket rejected.')


    return redirect('admin_ticket_review')

@login_required
def submit_training_data(request):
    """Office staff submit training data directly (no tickets)"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        office = office_user.office
    except OfficeUser.DoesNotExist:
        messages.error(request, 'You are not associated with any office.')
        return redirect('chat_interface')
    
    if request.method == 'POST':
        try:
            entry_count = 0
            created_entries = []
            index = 1
            
            # Loop through entries
            while f'answer_{index}' in request.POST:
                category_id = request.POST.get(f'category_{index}')
                new_category_name = request.POST.get(f'new_category_name_{index}', '').strip()
                new_category_description = request.POST.get(f'new_category_description_{index}', '').strip()
                
                intent_id = request.POST.get(f'intent_{index}')
                new_intent_name = request.POST.get(f'new_intent_name_{index}', '').strip()
                new_intent_description = request.POST.get(f'new_intent_description_{index}', '').strip()
                answer = request.POST.get(f'answer_{index}', '').strip()
                notes = request.POST.get(f'notes_{index}', '').strip()
                
                # Get questions array
                questions_list = request.POST.getlist(f'questions_{index}[]')
                questions_list = [q.strip() for q in questions_list if q.strip()]
                
                # Validation
                if not questions_list:
                    messages.error(request, f'Entry {index}: Please provide at least one question.')
                    return redirect('submit_training_data')
                
                if not answer:
                    messages.error(request, f'Entry {index}: Answer is required.')
                    return redirect('submit_training_data')
                
                # Handle category (existing or new)
                category = None
                if category_id == '__new__' and new_category_name:
                    # Create new category
                    category, created = Category.objects.get_or_create(
                        name=new_category_name,
                        defaults={
                            'description': new_category_description,
                            'created_by': request.user,
                            'is_active': True
                        }
                    )
                    
                    # Add to available categories for other entries
                    if created:
                        # Category was just created, add to the list for this session
                        pass
                    
                elif category_id:
                    try:
                        category = Category.objects.get(id=category_id)
                    except Category.DoesNotExist:
                        messages.error(request, f'Entry {index}: Invalid category.')
                        return redirect('submit_training_data')
                else:
                    messages.error(request, f'Entry {index}: Please select or create a category.')
                    return redirect('submit_training_data')
                
                # Get or create intent
                intent = None
                if new_intent_name:
                    intent, created = Intent.objects.get_or_create(
                        name=new_intent_name,
                        defaults={
                            'description': new_intent_description,
                            'category': category
                        }
                    )
                    if not created and intent.category != category:
                        # Update category if intent exists
                        intent.category = category
                        intent.save()
                elif intent_id:
                    try:
                        intent = Intent.objects.get(id=intent_id)
                        # Update category
                        intent.category = category
                        intent.save()
                    except Intent.DoesNotExist:
                        messages.error(request, f'Entry {index}: Selected intent not found.')
                        return redirect('submit_training_data')
                else:
                    messages.error(request, f'Entry {index}: Please select an intent or create a new one.')
                    return redirect('submit_training_data')
                
                # Check if training data exists for this intent
                existing = TrainingData.objects.filter(intent=intent, is_active=True).first()
                
                if existing:
                    # Replace existing questions
                    existing.set_questions(questions_list)
                    existing.answer = answer
                    existing.is_reviewed = False
                    existing.submitted_by = request.user
                    existing.office = office
                    existing.submitted_at = timezone.now()
                    existing.notes = notes
                    existing.save()
                    
                    created_entries.append(f"{intent.name} (updated with {len(questions_list)} questions)")
                    entry_count += 1
                else:
                    # Create new training data
                    training_data = TrainingData.objects.create(
                        intent=intent,
                        answer=answer,
                        submitted_by=request.user,
                        office=office,
                        is_reviewed=False,
                        notes=notes,
                        is_active=True
                    )
                    training_data.set_questions(questions_list)
                    training_data.save()
                    
                    created_entries.append(f"{intent.name} ({len(questions_list)} questions)")
                    entry_count += 1
                
                index += 1
            
            if entry_count > 0:
                messages.success(
                    request,
                    f'✓ Successfully submitted {entry_count} training data entries! '
                    f'Entries: {", ".join(created_entries)}. '
                    f'An admin will review and retrain the model to activate your submissions.'
                )
            else:
                messages.warning(request, 'No new entries were submitted.')
            
            return redirect('office_dashboard')
            
        except Exception as e:
            messages.error(request, f'Error submitting training data: {str(e)}')
            import traceback
            print(traceback.format_exc())
            return redirect('submit_training_data')
    
    # GET request
    intents = Intent.objects.all().select_related('category').order_by('name')
    categories = Category.objects.filter(is_active=True).order_by('order', 'name')
    
    # Get training data for each intent
    intents_with_data = []
    for intent in intents:
        training = TrainingData.objects.filter(intent=intent, is_active=True).first()
        intent_data = {
            'id': intent.id,
            'name': intent.name,
            'description': intent.description,
            'category_id': intent.category.id if intent.category else None,
            'has_training': training is not None,
        }
        
        if training:
            intent_data['questions'] = training.get_questions()
            intent_data['answer'] = training.answer
        
        intents_with_data.append(intent_data)
    
    categories_data = [
        {
            'id': cat.id,
            'name': cat.name,
            'description': cat.description
        }
        for cat in categories
    ]
    
    intents_json = json.dumps(intents_with_data)
    categories_json = json.dumps(categories_data)
    
    return render(request, 'submit_training_data.html', {
        'office': office,
        'intents_json': intents_json,
        'categories_json': categories_json,
    })


@login_required
def review_training_data(request):
    """Admin review page for training data submissions"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')
    
    filter_type = request.GET.get('filter', 'unreviewed')
    
    submissions_list = TrainingData.objects.filter(
        is_active=True,
        submitted_by__isnull=False
    ).select_related('intent', 'submitted_by', 'office', 'reviewed_by')
    
    if filter_type == 'unreviewed':
        submissions_list = submissions_list.filter(is_reviewed=False)
    elif filter_type == 'reviewed':
        submissions_list = submissions_list.filter(is_reviewed=True)
    
    submissions_list = submissions_list.order_by('is_reviewed', '-submitted_at')
    
    # Stats
    total_count = TrainingData.objects.filter(is_active=True, submitted_by__isnull=False).count()
    unreviewed_count = TrainingData.objects.filter(is_active=True, submitted_by__isnull=False, is_reviewed=False).count()
    reviewed_count = TrainingData.objects.filter(is_active=True, submitted_by__isnull=False, is_reviewed=True).count()
    
    # Pagination
    paginator = Paginator(submissions_list, 20)
    page_number = request.GET.get('page')
    submissions = paginator.get_page(page_number)
    
    return render(request, 'review_training_data.html', {
        'submissions': submissions,
        'filter': filter_type,
        'total_count': total_count,
        'unreviewed_count': unreviewed_count,
        'reviewed_count': reviewed_count,
    })


@login_required
@require_http_methods(["POST"])
def mark_training_reviewed(request, training_id):
    """Mark training data as reviewed"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    try:
        training = TrainingData.objects.get(id=training_id)
        training.mark_as_reviewed(request.user)
        
        return JsonResponse({'success': True})
    except TrainingData.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Training data not found'})


@login_required
def view_training_detail(request, training_id):
    """View detailed training data (for review)"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('chat_interface')
    
    training = get_object_or_404(TrainingData, id=training_id)
    questions_list = training.get_questions()
    
    return render(request, 'view_training_detail.html', {
        'training': training,
        'questions_list': questions_list,
    })
    
    
@login_required
def submit_location(request):
    """Office staff submit locations directly (no tickets)"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        office = office_user.office
    except OfficeUser.DoesNotExist:
        messages.error(request, 'You are not associated with any office.')
        return redirect('chat_interface')
    
    if request.method == 'POST':
        try:
            entry_count = 0
            created_entries = []
            skipped_entries = []
            index = 1
            
            print("\n" + "="*80)
            print("LOCATION SUBMISSION DEBUG")
            print("="*80)
            
            # Loop through entries
            while f'room_number_{index}' in request.POST:
                print(f"\n--- Processing Location {index} ---")
                
                room_number = request.POST.get(f'room_number_{index}', '').strip()
                room_name = request.POST.get(f'room_name_{index}', '').strip()
                building = request.POST.get(f'building_{index}', '').strip()
                floor = request.POST.get(f'floor_{index}', '').strip()
                description = request.POST.get(f'description_{index}', '').strip()
                
                # Get keywords array
                keywords_list = request.POST.getlist(f'keywords_{index}[]')
                keywords_list = [k.strip().lower() for k in keywords_list if k.strip()]
                
                # DEBUG OUTPUT
                print(f"Room Number: {room_number}")
                print(f"Room Name: {room_name}")
                print(f"Building: {building}")
                print(f"Floor: {floor}")
                print(f"Keywords: {keywords_list}")
                
                # Validation
                if not room_number or not building or not floor:
                    messages.error(request, f'Location {index}: Room number, building, and floor are required.')
                    print(f"❌ ERROR: Missing required fields")
                    return redirect('submit_location')
                
                if not keywords_list:
                    messages.error(request, f'Location {index}: Please provide at least one keyword.')
                    print(f"❌ ERROR: No keywords provided")
                    return redirect('submit_location')
                
                # Check if location already exists
                if Location.objects.filter(room_number=room_number).exists():
                    skipped_entries.append(f"{room_number} (already exists)")
                    print(f"⚠️  Location {room_number} already exists, skipping")
                    index += 1
                    continue
                
                # Create location
                location = Location.objects.create(
                    room_number=room_number,
                    room_name=room_name,
                    building=building,
                    floor=floor,
                    description=description,
                    is_active=True
                )
                
                print(f"✅ Created Location (ID: {location.id})")
                
                # Always add room number as first keyword
                if room_number.lower() not in keywords_list:
                    keywords_list.insert(0, room_number.lower())
                
                # Create keywords
                keyword_count = 0
                for keyword in keywords_list:
                    LocationKeyword.objects.create(
                        location=location,
                        keyword=keyword,
                        priority=1
                    )
                    keyword_count += 1
                
                print(f"✅ Created {keyword_count} keywords")
                print(f"   Keywords: {keywords_list}")
                
                created_entries.append(f"{room_number} ({keyword_count} keywords)")
                entry_count += 1
                
                index += 1
            
            # Reload location extractor
            if entry_count > 0:
                try:
                    from .hybrid_predictor import HybridChatbotPredictor
                    predictor = HybridChatbotPredictor()
                    predictor.location_extractor._load_locations()
                    print(f"✅ Reloaded location extractor")
                except Exception as e:
                    print(f"⚠️  Warning: Could not reload location extractor: {e}")
            
            print("\n" + "="*80)
            print(f"✅ SUBMISSION COMPLETE: {entry_count} locations created")
            if skipped_entries:
                print(f"⚠️  SKIPPED: {len(skipped_entries)} locations")
            print("="*80 + "\n")
            
            # Success message
            if entry_count > 0:
                success_msg = f'✓ Successfully submitted {entry_count} location(s)! Entries: {", ".join(created_entries)}. '
                if skipped_entries:
                    success_msg += f'Skipped {len(skipped_entries)} duplicate(s): {", ".join(skipped_entries)}. '
                success_msg += 'Locations are now available in the chatbot immediately!'
                messages.success(request, success_msg)
            elif skipped_entries:
                messages.warning(request, f'All {len(skipped_entries)} location(s) already exist: {", ".join(skipped_entries)}')
            else:
                messages.warning(request, 'No locations were submitted.')
            
            return redirect('office_dashboard')
            
        except Exception as e:
            messages.error(request, f'Error submitting locations: {str(e)}')
            import traceback
            print("\n" + "="*80)
            print("❌ ERROR IN SUBMISSION")
            print("="*80)
            print(traceback.format_exc())
            print("="*80 + "\n")
            return redirect('submit_location')
    
    # GET request - Get all existing locations
    existing_locations = Location.objects.filter(
        is_active=True
    ).prefetch_related('keywords').order_by('building', 'floor', 'room_number')
    
    return render(request, 'submit_location.html', {
        'office': office,
        'existing_locations': existing_locations,
    })
    

@login_required
def office_view_all_training(request):
    """Office staff view all their training data submissions"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        office = office_user.office
    except OfficeUser.DoesNotExist:
        messages.error(request, 'You are not associated with any office.')
        return redirect('chat_interface')
    
    training_data_list = TrainingData.objects.filter(
        office=office,
        submitted_by=request.user
    ).select_related('intent', 'reviewed_by').order_by('-submitted_at')
    
    # Calculate statistics
    pending_count = training_data_list.filter(is_reviewed=False).count()
    reviewed_count = training_data_list.filter(is_reviewed=True).count()
    total_questions = sum(len(t.get_questions()) for t in training_data_list)
    
    # Pagination
    paginator = Paginator(training_data_list, 5)  # 5 items per page
    page_number = request.GET.get('page')
    
    try:
        training_data = paginator.page(page_number)
    except PageNotAnInteger:
        training_data = paginator.page(1)
    except EmptyPage:
        training_data = paginator.page(paginator.num_pages)
    
    return render(request, 'office_view_all_training.html', {
        'office': office,
        'training_data': training_data,
        'pending_count': pending_count,
        'reviewed_count': reviewed_count,
        'total_questions': total_questions,
    })


@login_required
def office_view_training(request, training_id):
    """Office staff view specific training data detail"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        office = office_user.office
    except OfficeUser.DoesNotExist:
        messages.error(request, 'You are not associated with any office.')
        return redirect('chat_interface')
    
    training = get_object_or_404(TrainingData, id=training_id)
    
    # Check if user has access
    if training.office != office and not request.user.is_staff:
        messages.error(request, 'Access denied.')
        return redirect('office_dashboard')
    
    questions_list = training.get_questions()
    
    return render(request, 'office_view_training.html', {
        'office': office,
        'training': training,
        'questions_list': questions_list,
    })


@login_required
def office_view_all_locations(request):
    """Office staff view all locations"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        office = office_user.office
    except OfficeUser.DoesNotExist:
        messages.error(request, 'You are not associated with any office.')
        return redirect('chat_interface')
    
    locations_list = Location.objects.filter(
        is_active=True
    ).prefetch_related('keywords').order_by('-created_at')
    
    # Pagination
    paginator = Paginator(locations_list, 5)  # 5 items per page
    page_number = request.GET.get('page')
    
    try:
        locations = paginator.page(page_number)
    except PageNotAnInteger:
        locations = paginator.page(1)
    except EmptyPage:
        locations = paginator.page(paginator.num_pages)
    
    return render(request, 'office_view_all_locations.html', {
        'office': office,
        'locations': locations,
    })


@login_required
def office_view_location(request, location_id):
    """Office staff view specific location detail"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
        office = office_user.office
    except OfficeUser.DoesNotExist:
        messages.error(request, 'You are not associated with any office.')
        return redirect('chat_interface')
    
    location = get_object_or_404(Location, id=location_id)
    keywords = [kw.keyword for kw in location.keywords.all()]
    
    return render(request, 'office_view_location.html', {
        'office': office,
        'location': location,
        'keywords': keywords,
    })
    
@login_required
def office_location_detail(request, location_id):
    """Get location details for editing (AJAX)"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
    except OfficeUser.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Not authorized'}, status=403)
    
    try:
        location = Location.objects.get(id=location_id)
        keywords = [kw.keyword for kw in location.keywords.all()]
        
        return JsonResponse({
            'success': True,
            'location': {
                'id': location.id,
                'room_number': location.room_number,
                'room_name': location.room_name or '',
                'building': location.building,
                'floor': location.floor,
                'description': location.description or '',
                'keywords': keywords
            }
        })
    except Location.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Location not found'}, status=404)


@login_required
def office_edit_location(request, location_id):
    """Office staff edit location"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
    except OfficeUser.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Not authorized'}, status=403)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            location = Location.objects.get(id=location_id)
            
            # Update location
            location.room_number = data.get('room_number')
            location.room_name = data.get('room_name', '')
            location.building = data.get('building')
            location.floor = data.get('floor')
            location.description = data.get('description', '')
            location.save()
            
            # Update keywords
            location.keywords.all().delete()  # Remove old keywords
            
            keywords = data.get('keywords', [])
            for keyword in keywords:
                if keyword.strip():
                    LocationKeyword.objects.create(
                        location=location,
                        keyword=keyword.strip().lower(),
                        priority=1
                    )
            
            # Reload location extractor
            try:
                from .hybrid_predictor import HybridChatbotPredictor
                predictor = HybridChatbotPredictor()
                predictor.location_extractor._load_locations()
            except Exception as e:
                print(f"Warning: Could not reload location extractor: {e}")
            
            return JsonResponse({'success': True})
            
        except Location.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Location not found'}, status=404)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)


@login_required
def office_delete_location(request, location_id):
    """Office staff delete location"""
    try:
        office_user = OfficeUser.objects.get(user=request.user)
    except OfficeUser.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Not authorized'}, status=403)
    
    if request.method == 'POST':
        try:
            location = Location.objects.get(id=location_id)
            
            # Delete location (keywords will be deleted automatically via CASCADE)
            location.delete()
            
            # Reload location extractor
            try:
                from .hybrid_predictor import HybridChatbotPredictor
                predictor = HybridChatbotPredictor()
                predictor.location_extractor._load_locations()
            except Exception as e:
                print(f"Warning: Could not reload location extractor: {e}")
            
            return JsonResponse({'success': True})
            
        except Location.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Location not found'}, status=404)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
    
    return JsonResponse({'success': False, 'error': 'Invalid method'}, status=405)