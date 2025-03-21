from django import forms
from .models import Document, Query


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('title', 'file')


class QueryForm(forms.ModelForm):
    class Meta:
        model = Query
        fields = ('question',)
        widgets = {
            'question': forms.TextInput(
                attrs={'class': 'form-control', 'placeholder': 'Ask a question about the document'})
        }
