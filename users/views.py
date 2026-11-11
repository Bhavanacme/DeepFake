from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegistrationForm, UserLoginForm, UserProfileForm, UserProfileDetailsForm
from .models import UserProfile


def register_view(request):
    """
    Handles user registration with validation, password confirmation, and profile generation.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"Welcome {user.first_name or user.username}! Your account has been created. Please log in.")
            return redirect('login')
        else:
            messages.error(request, "Please correct the errors in the form below.")
    else:
        form = UserRegistrationForm()

    context = {
        'form': form,
        'page_title': 'Create Account | Deepfake Detection System'
    }
    return render(request, 'users/register.html', context)


def login_view(request):
    """
    Handles user authentication supporting both username and email login.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username_or_email = form.cleaned_data.get('username').strip()
            password = form.cleaned_data.get('password')

            # Check if user entered an email instead of username
            username_to_auth = username_or_email
            if '@' in username_or_email:
                user_match = User.objects.filter(email__iexact=username_or_email).first()
                if user_match:
                    username_to_auth = user_match.username

            user = authenticate(request, username=username_to_auth, password=password)
            if user is not None:
                if user.is_active:
                    login(request, user)
                    messages.success(request, f"Welcome back, {user.first_name or user.username}!")
                    next_url = request.GET.get('next')
                    return redirect(next_url if next_url else 'dashboard')
                else:
                    messages.error(request, "This account is currently disabled.")
            else:
                messages.error(request, "Invalid username/email or password. Please try again.")
    else:
        form = UserLoginForm()

    context = {
        'form': form,
        'page_title': 'User Login | Deepfake Detection System'
    }
    return render(request, 'users/login.html', context)


def logout_view(request):
    """
    Logs out the current authenticated user and redirects to the homepage.
    """
    if request.user.is_authenticated:
        logout(request)
        messages.info(request, "You have been logged out successfully.")
    return redirect('home')


@login_required
def profile_view(request):
    """
    Displays and handles user profile details and statistics.
    """
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)

    if request.method == 'POST':
        user_form = UserProfileForm(request.POST, instance=user)
        profile_form = UserProfileDetailsForm(request.POST, request.FILES, instance=profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, "Your profile details have been successfully updated.")
            return redirect('profile')
        else:
            messages.error(request, "Please check the form for errors.")
    else:
        user_form = UserProfileForm(instance=user)
        profile_form = UserProfileDetailsForm(instance=profile)

    # Statistics for the profile overview
    total_detections = user.detection_results.count()
    real_count = user.detection_results.filter(result='REAL').count()
    fake_count = user.detection_results.filter(result='FAKE').count()

    context = {
        'user_form': user_form,
        'profile_form': profile_form,
        'profile': profile,
        'total_detections': total_detections,
        'real_count': real_count,
        'fake_count': fake_count,
        'page_title': 'User Profile | Deepfake Detection System'
    }
    return render(request, 'users/profile.html', context)
