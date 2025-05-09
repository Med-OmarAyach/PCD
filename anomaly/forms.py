from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

class CustomUserCreationForm(forms.ModelForm):
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput,
        min_length=8,
        help_text="At least 8 characters"
    )
    
    class Meta:
        model = CustomUser
        fields = ['email', 'first_name', 'last_name', 'password']