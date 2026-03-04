from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    
    # Public
    path('', views.chat_interface, name='chat_interface'),
    path('chat/api/', views.chat_api, name='chat_api'),
    path('chat/menu-data/', views.get_menu_categories, name='get_menu_categories'),
    
    # Office Staff
    path('office/', views.office_dashboard, name='office_dashboard'),
    path('office/ticket/<int:ticket_id>/', views.ticket_detail, name='ticket_detail'),
    
    # Admin
    path('admin-panel/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-panel/add/', views.add_training_data, name='add_training_data'),
    path('admin-panel/train/', views.train_model, name='train_model'),
    path('admin-panel/tickets/', views.admin_ticket_review, name='admin_ticket_review'),
    path('admin-panel/tickets/<int:ticket_id>/approve/', views.approve_ticket, name='approve_ticket'),
    path('admin-panel/tickets/<int:ticket_id>/reject/', views.reject_ticket, name='reject_ticket'),
    path('admin-panel/tickets/bulk-approve/', views.bulk_approve_tickets, name='bulk_approve_tickets'),
    
    # Location tickets
    path('office/submit-location-ticket/', views.submit_location_ticket, name='submit_location_ticket'),
    
    # Location ticket detail (Office staff)
    path('office/location-ticket/<int:ticket_id>/', views.location_ticket_detail, name='location_ticket_detail'),
    path('office/location-ticket/<int:ticket_id>/comment/', views.add_location_ticket_comment, name='add_location_ticket_comment'),
    # Location ticket admin actions
    path('admin-panel/location-tickets/<int:ticket_id>/approve/', views.approve_location_ticket, name='approve_location_ticket'),
    path('admin-panel/locations/<int:location_id>/update/', views.update_location, name='update_location'),
    path('admin-panel/location-tickets/<int:ticket_id>/reject/', views.reject_location_ticket, name='reject_location_ticket'),
    path('admin-panel/location-tickets/bulk-approve/', views.bulk_approve_location_tickets, name='bulk_approve_location_tickets'),
    
    # AJAX endpoints
    path('ajax/create-intent/', views.create_intent_ajax, name='create_intent_ajax'),
    path('ajax/training-data/', views.get_training_data_ajax, name='get_training_data_ajax'),
    path('ajax/conversations/', views.get_conversations_ajax, name='get_conversations_ajax'),
    
    # AJAX endpoints for Training Data CRUD
    path('ajax/training-data/create/', views.create_training_data_ajax, name='create_training_data_ajax'),
    path('ajax/training-data/<int:training_id>/', views.get_training_data_detail_ajax, name='get_training_data_detail_ajax'),
    path('ajax/training-data/<int:training_id>/update/', views.update_training_data_ajax, name='update_training_data_ajax'),
    path('ajax/training-data/<int:training_id>/delete/', views.delete_training_data_ajax, name='delete_training_data_ajax'),
    
    # Admin - Location management
    path('admin-panel/locations/', views.manage_locations, name='manage_locations'),
    path('admin-panel/location/<int:location_id>/detail/', views.admin_location_detail, name='admin_location_detail'),
    path('admin-panel/location/<int:location_id>/update/', views.update_location, name='update_location'),
    path('admin-panel/location/<int:location_id>/delete/', views.delete_location, name='delete_location'),
    
    path('submit-feedback/', views.submit_feedback, name='submit_feedback'),
    
    # Feedback management
    path('admin-panel/feedback/', views.manage_feedback, name='manage_feedback'),
    path('admin-panel/feedback/<int:feedback_id>/resolve/', views.resolve_feedback, name='resolve_feedback'),
    path('admin-panel/feedback/<int:feedback_id>/details/', views.feedback_details, name='feedback_details'),
    path('admin-panel/feedback/<int:feedback_id>/notes/', views.save_feedback_notes, name='save_feedback_notes'),
    path('admin-panel/feedback/<int:feedback_id>/delete/', views.delete_feedback, name='delete_feedback'),
    
    # User management
    path('admin-panel/users/', views.manage_users, name='manage_users'),
    path('admin-panel/users/create/', views.register_user, name='register_user'),
    path('admin-panel/users/<int:user_id>/update/',views.update_user_ajax,name='update_user_ajax'),
    path('admin-panel/users/<int:user_id>/delete/', views.delete_user, name='delete_user'),
    
    # New simplified training data submission
    path('office/submit-training-data/', views.submit_training_data, name='submit_training_data'),
    path('admin-panel/review-training-data/', views.review_training_data, name='review_training_data'),
    path('admin-panel/training-data/<int:training_id>/mark-reviewed/', views.mark_training_reviewed, name='mark_training_reviewed'),
    path('admin-panel/training-data/<int:training_id>/', views.view_training_detail, name='view_training_detail'),
    
    # Direct location submission (no tickets)
    path('office/submit-location/', views.submit_location, name='submit_location'),
    
    # Office staff viewing
    path('office/training/', views.office_view_all_training, name='office_view_all_training'),
    path('office/training/<int:training_id>/', views.office_view_training, name='office_view_training'),
    path('office/locations/', views.office_view_all_locations, name='office_view_all_locations'),
    path('office/locations/<int:location_id>/', views.office_view_location, name='office_view_location'),
    
    # Office staff - Location management
    path('office/location/<int:location_id>/detail/', views.office_location_detail, name='office_location_detail'),
    path('office/location/<int:location_id>/edit/', views.office_edit_location, name='office_edit_location'),
    path('office/location/<int:location_id>/delete/', views.office_delete_location, name='office_delete_location'),
    
    # Category management
    path('admin-panel/categories/', views.manage_categories, name='manage_categories'),
    path('admin-panel/categories/<int:category_id>/update/', views.update_category, name='update_category'),
    
    path('admin-panel/categories/<int:category_id>/delete/', views.delete_category, name='delete_category'),

    path('admin-panel/intents/create/', views.create_intent, name='create_intent'),
    path('admin-panel/intents/<int:intent_id>/update/', views.update_intent, name='update_intent'),
    path('admin-panel/intents/<int:intent_id>/delete/', views.delete_intent, name='delete_intent'),
    
    path('admin-panel/offices/create/', views.create_office, name='create_office'),
    path('admin-panel/offices/<int:office_id>/update/', views.update_office, name='update_office'),
    path('admin-panel/offices/<int:office_id>/delete/', views.delete_office, name='delete_office'),
]