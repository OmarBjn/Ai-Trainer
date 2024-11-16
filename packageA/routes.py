from packageA.models import User
from packageA import testBicepcurl, testPushup
from flask import render_template, url_for, flash, redirect, request, send_file
from packageA.forms import RegistrationForm, LoginForm
from packageA import app, bcrypt, db
import os
from flask_login import (
    login_required,
    login_user,
    current_user,
    logout_user,
    login_required,
)


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@ app.route('/services', methods=['GET', 'POST'])
def services():
    return render_template('services.html')

@ app.route('/pushUp', methods=['GET', 'POST'])
def pushUp():
    return render_template('pushUp.html')

# Route to handle video upload and processing
@app.route('/processPushupVideo', methods=['POST'])
def processPushupVideo():

    # Get the uploaded file from the request
    file = request.files['file']

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Process the uploaded video
    processed_video_path = testPushup.processVideo(file_path)

    # Return the processed video
    return send_file(processed_video_path, mimetype='video/mp4')

@ app.route('/bicepcurl', methods=['GET', 'POST'])
def bicepcurl():
    return render_template('bicepcurl.html')

# Route to handle video upload and processing
@app.route('/processBicepcurlVideo', methods=['POST'])
def processBicepcurlVideo():

    # Get the uploaded file from the request
    file = request.files['file']

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Process the uploaded video
    processed_video_path = testBicepcurl.processVideo(file_path)

    # Return the processed video
    return send_file(processed_video_path, mimetype='video/mp4')

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode(
            "utf-8"
        )
        user = User(
            fname=form.fname.data,
            lname=form.lname.data,
            username=form.username.data,
            email=form.email.data,
            password=hashed_password,
        )
        db.session.add(user)
        db.session.commit()
        flash(f"Account created successfully for {form.username.data}", "success")
        return redirect(url_for("login"))
    return render_template("register.html", title="Register", form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            flash("You have been logged in!", "success")
            return redirect(next_page) if next_page else redirect(url_for("home"))
        else:
            flash("Login Unsuccessful. Please check credentials", "danger")
    return render_template("login.html", title="Login", form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("home"))