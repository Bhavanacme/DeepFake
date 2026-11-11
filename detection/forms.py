from django import forms
from .preprocessing import validate_uploaded_media


class MediaUploadForm(forms.Form):
    """
    Form for validating and accepting image and video uploads for deepfake analysis.
    """
    file = forms.FileField(
        required=True,
        widget=forms.FileInput(attrs={
            'id': 'mediaFileInput',
            'class': 'd-none',
            'accept': '.jpg,.jpeg,.png,.mp4,.avi,.mov,.mkv'
        }),
        error_messages={
            'required': 'Please select an image or video file to analyze.'
        }
    )

    def clean_file(self):
        uploaded_file = self.cleaned_data.get('file')
        is_valid, err_msg, media_type = validate_uploaded_media(uploaded_file)
        if not is_valid:
            raise forms.ValidationError(err_msg)
        return uploaded_file
