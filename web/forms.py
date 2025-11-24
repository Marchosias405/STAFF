from django import forms

class HSPredictionForm(forms.Form):
    tags = forms.CharField(
        label="Product Tags",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'laptop, electronics, computer'
        }),
        help_text="Comma-separated tags"
    )
    
    price = forms.FloatField(
        label="Price (USD)",
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '0.00',
            'step': '0.01'
        })
    )
    
    weight = forms.FloatField(
        label="Weight (kg)",
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '0.00',
            'step': '0.01'
        })
    )
    
    origin = forms.CharField(
        label="Origin Country",
        max_length=10,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'US'
        })
    )
    
    dest = forms.CharField(
        label="Destination Country",
        max_length=10,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'CA'
        })
    )
    
    gift = forms.BooleanField(
        label="Is this a gift?",
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )
    
    topk = forms.IntegerField(
        label="Number of Predictions",
        initial=5,
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control'
        })
    )