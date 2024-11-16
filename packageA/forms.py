from tokenize import String
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import (
    DataRequired,
    Length,
    Email,
    Regexp,
    EqualTo,
    ValidationError,
)
from packageA.models import User


class RegistrationForm(FlaskForm):
    fname = StringField(
        "First Name", validators=[DataRequired(), Length(min=2, max=25)], render_kw={"placeholder": "Enter your first name"}
    )
    lname = StringField("Last Name", validators=[DataRequired(), Length(min=2, max=25)],render_kw={"placeholder": "Enter your last name"})
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=2, max=25)], render_kw={"placeholder": "Enter your username"}
    )
    email = StringField("Email", validators=[DataRequired(), Email()], render_kw={"placeholder": "Enter your Email"})
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Regexp(
                "^.{8,}$",
                message="at least 8 characters long."
            ),
            
        ],
        render_kw={"placeholder": "Enter your password"}
    )
    confirm_password = PasswordField(
        "Confirm Password", validators=[DataRequired(), EqualTo("password")], render_kw={"placeholder": "Enter your password again"}
    )
    submit = SubmitField("Sign Up")

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError(
                "Username already exists!"
            )
        
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError(
                "Email already exists!"
            )
        
class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()], render_kw={"placeholder": "Enter your Email"})
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
        ],
        render_kw={"placeholder": "Enter your password"}
    )
    remember = BooleanField("Remember Me")
    submit = SubmitField("Log In")
