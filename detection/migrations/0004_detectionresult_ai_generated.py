"""
Migration 0004: Add AI-Generated image detection fields to DetectionResult.

Changes:
  - result field: adds 'AI_GENERATED' as a valid choice (max_length stays 20)
  - ai_generated_probability (FloatField, default=0.0)
  - ai_gen_model_name (CharField, max_length=150, blank=True, default='')
  - ai_gen_model_available (BooleanField, default=False)
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detection', '0003_detectionresult_calibration_status_and_more'),
    ]

    operations = [
        # Update the result field choices to include AI_GENERATED
        migrations.AlterField(
            model_name='detectionresult',
            name='result',
            field=models.CharField(
                default='UNCERTAIN',
                max_length=20,
                choices=[
                    ('REAL', 'Real / Authentic'),
                    ('FAKE', 'Fake / Deepfake Detected'),
                    ('AI_GENERATED', 'AI-Generated Image'),
                    ('UNCERTAIN', 'Inconclusive / Low Quality'),
                ],
            ),
        ),
        # New field: probability score from AI-gen detector
        migrations.AddField(
            model_name='detectionresult',
            name='ai_generated_probability',
            field=models.FloatField(
                default=0.0,
                help_text='Probability of image being fully AI-generated (0.0 - 1.0)',
            ),
        ),
        # New field: which AI-gen model was used
        migrations.AddField(
            model_name='detectionresult',
            name='ai_gen_model_name',
            field=models.CharField(
                blank=True,
                default='',
                max_length=150,
                help_text='Name of the AI-generated image detection model used.',
            ),
        ),
        # New field: was the AI-gen model available during this scan?
        migrations.AddField(
            model_name='detectionresult',
            name='ai_gen_model_available',
            field=models.BooleanField(
                default=False,
                help_text='Whether the AI-gen detection model was loaded and run.',
            ),
        ),
    ]
