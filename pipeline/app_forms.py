import os
from pathlib import Path

from flask_wtf import FlaskForm
from wtforms.fields import SubmitField, DateField, TimeField, SelectMultipleField, StringField
from wtforms.widgets import ListWidget, CheckboxInput
from wtforms.validators import DataRequired, ValidationError

from carbonpipeline.Processing.constants import VARIABLES_FOR_PREDICTOR

# ================== AmeriFlux predictors and gas flux ==================

PREDICTORS = sorted(VARIABLES_FOR_PREDICTOR.keys())

# =======================================================================

class AreaTypeForm(FlaskForm):
    start_date = DateField(format='%Y-%m-%d', validators=[DataRequired()])
    end_date = DateField(format='%Y-%m-%d', validators=[DataRequired()])

    start_time = TimeField(format='%H:%M', validators=[DataRequired()])
    end_time = TimeField(format='%H:%M', validators=[DataRequired()])

    predictors = SelectMultipleField(
        'Predictors',
        choices=[(p, p) for p in PREDICTORS],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False),
    )

    submit = SubmitField('Submit')

    def validate_predictors(self, field):
        if not field.data:
            raise ValidationError('Select at least one predictor.')


class AreaTypePointForm(AreaTypeForm):
    latitude = StringField('Coordinates', validators=[DataRequired()])
    longitude = StringField(validators=[DataRequired()])
    csv = StringField('CSV', validators=[DataRequired()])

    def validate_csv(self, field):
        if not os.path.exists(Path(field.data)):
            raise ValidationError('The relative path to the CSV is invalid.')
        if not field.data.endswith('.csv'):
            raise ValidationError('A CSV file is required.')


class AreaTypeBoundingBoxForm(AreaTypeForm):
    north = StringField('Coordinates')
    east = StringField()
    south = StringField()
    west = StringField()


class AreaTypePolygonsForm(AreaTypeForm):
    geojsons = StringField('GeoJSONs', validators=[DataRequired()])
    rid = StringField('Relative ID (optional)')

    def validate_geojsons(self, field):
        if not os.path.exists(Path(field.data)):
            raise ValidationError('The relative path to the GeoJSONs folder is invalid.')


class CredentialsForm(FlaskForm):
    account = StringField('Account', validators=[DataRequired()])

    submit = SubmitField('Submit')