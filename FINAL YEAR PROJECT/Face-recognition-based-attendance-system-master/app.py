import cv2
import os
from flask import Flask,request,render_template, jsonify, session, redirect, url_for, flash
from datetime import date, timedelta, datetime
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import calendar

load_dotenv()

#### Defining Flask App
app = Flask(__name__)

# Set secret key
app.secret_key = 'your-secret-key-here'

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if img is None:
        return []
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    except Exception as e:
        print(f"Error in extract_faces: {e}")
        return []

#### Identify face using ML model
def identify_face(facearray):
    try:
        # Load the pre-trained model
        model = joblib.load('static/face_recognition_model.pkl')
        
        # Get the probabilities for each class
        probabilities = model.predict_proba(facearray)
        
        # Get the class with the highest probability
        max_prob_index = np.argmax(probabilities)
        max_prob = probabilities[0][max_prob_index]
        identified_person = model.classes_[max_prob_index]
        
        # Set a threshold for the confidence level
        confidence_threshold = 0.7  # Increased threshold for more confidence
        print(f"Confidence: {max_prob:.2f} for {identified_person}")
        
        # If the highest probability is less than the threshold, return "Unknown"
        if max_prob < confidence_threshold:
            return ["Unknown"]
        else:
            return [identified_person]
            
    except Exception as e:
        print(f"Error in identify_face: {e}")
        return ["Unknown"]



#### A function which trains the model on all the faces available in faces folder
def train_model():
    try:
        faces = []
        labels = []
        userlist = os.listdir('static/faces')
        
        if not userlist:
            print("No users found in faces directory")
            return False
            
        for user in userlist:
            user_images = os.listdir(f'static/faces/{user}')
            if not user_images:
                continue
                
            for imgname in user_images:
                img = cv2.imread(f'static/faces/{user}/{imgname}')
                if img is None:
                    continue
                    
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
        
        if not faces:
            print("No valid face images found")
            return False
            
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        
        # Save the model with protocol=4 to ensure compatibility
        joblib.dump(knn, 'static/face_recognition_model.pkl', protocol=4)
        print(f"Model trained successfully with {len(faces)} images")
        return True
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        return False


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add_attendance(name):
    # Check if the name is "Unknown"
    if name == "Unknown":
        print("Unknown person, attendance not recorded.")
        return
    
    # Validate the name format (should contain '_')
    if '_' not in name:
        print(f"Invalid name format: {name}. Attendance not recorded.")
        return

    try:
        # Extract username and userid
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Read the existing attendance CSV
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        
        # Add attendance only if userid is not already in the CSV
        if userid not in list(df['Roll'].astype(str)):  # Convert Roll to string for comparison
            with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
            print(f"Attendance recorded for {username} (ID: {userid}) at {current_time}.")
            return True
        else:
            print(f"Attendance for {username} (ID: {userid}) already recorded.")
            return False
    
    except Exception as e:
        print(f"Error processing attendance for '{name}': {e}")
        return False


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()    
    return render_template('home.html', 
                         names=names, 
                         rolls=rolls, 
                         times=times, 
                         l=l, 
                         totalreg=totalreg(), 
                         datetoday2=datetoday2,
                         logged_in='user' in session)

@app.route('/start', methods=['GET', 'POST'])
def start():
    if request.method == 'POST':
        # Check if the current time is within the allowed attendance period
        current_time = datetime.now().time()
        cutoff_time = datetime.strptime("20:00:00", "%H:%M:%S").time()
        
        if current_time > cutoff_time:
            return jsonify({
                'success': False,
                'message': 'Attendance can only be marked before 09:00 AM'
            })
        
        # Check if the trained model exists
        if 'face_recognition_model.pkl' not in os.listdir('static'):
            return jsonify({
                'success': False,
                'message': 'No trained model found. Please contact administrator.'
            })

        try:
            # Get the image from the request
            image_file = request.files['image']
            nparr = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Detect faces
            face_points = extract_faces(frame)
            if len(face_points) == 0:
                return jsonify({
                    'success': False,
                    'message': 'No face detected. Please try again.'
                })
            
            # Process the first detected face
            (x, y, w, h) = face_points[0]
            face_img = frame[y:y+h, x:x+w]
            face = cv2.resize(face_img, (50, 50))
            
            # Identify the person
            identified_person = identify_face(face.reshape(1, -1))[0]
            
            if identified_person == "Unknown":
                return jsonify({
                    'success': False,
                    'message': 'Unknown person. Please register first.'
                })
            
            # Mark attendance
            if add_attendance(identified_person):
                return jsonify({
                    'success': True,
                    'message': f'Attendance marked successfully for {identified_person.split("_")[0]}',
                    'reload': True
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Attendance already marked for today',
                    'reload': True
                })
            
        except Exception as e:
            print(f"Error processing attendance: {e}")
            return jsonify({
                'success': False,
                'message': 'Error processing attendance. Please try again.'
            })

    # GET request - just render the template
    return render_template('home.html',
                         names=extract_attendance()[0],
                         rolls=extract_attendance()[1],
                         times=extract_attendance()[2],
                         l=extract_attendance()[3],
                         totalreg=totalreg(),
                         datetoday2=datetoday2,
                         logged_in='user' in session)

# Function to get a list of registered users
def get_registered_users():
    # Fetch the list of registered users from the `static/faces` directory
    userlist = os.listdir('static/faces')
    return [user for user in userlist if os.path.isdir(f'static/faces/{user}')] 

#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

# Initialize users.csv if it doesn't exist
if 'users.csv' not in os.listdir():
    with open('users.csv', 'w') as f:
        f.write('email,password,role,first_name,last_name,roll_no,dob,mobile,address,join_date\n')

def init_admin():
    # Create admin user if not exists
    df = pd.read_csv('users.csv')
    if not any(df['email'] == 'admin@admin.com'):
        admin_data = {
            'email': 'admin@admin.com',
            'password': generate_password_hash('admin123'),
            'role': 'admin',
            'first_name': 'Admin',
            'last_name': 'User',
            'roll_no': 'ADMIN001',
            'dob': '2000-01-01',
            'mobile': '0000000000',
            'address': 'Admin Office',
            'join_date': datetime.now().strftime('%Y-%m-%d')
        }
        df = pd.concat([df, pd.DataFrame([admin_data])], ignore_index=True)
        df.to_csv('users.csv', index=False)

# Call init_admin when app starts
init_admin()

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session['role'] != 'admin':
            flash('Admin access required')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        df = pd.read_csv('users.csv')
        user = df[df['email'] == email]
        
        if not user.empty and check_password_hash(user['password'].iloc[0], password):
            session['user'] = email
            session['role'] = user['role'].iloc[0]
            flash(f'Welcome back, {user["first_name"].iloc[0]}!')
            return redirect(url_for('admin_dashboard' if session['role'] == 'admin' else 'user_dashboard'))
        
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'user' in session:
        flash('Logged out successfully')
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin/add_user', methods=['GET', 'POST'])
@admin_required
def add_user():
    if request.method == 'POST':
        new_user = {
            'email': request.form['email'],
            'password': generate_password_hash(request.form['password']),  # Use provided password instead of DOB
            'role': 'user',
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'roll_no': request.form['roll_no'],
            'dob': request.form['dob'],
            'mobile': request.form['mobile'],
            'address': request.form['address'],
            'join_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Validate password
        if len(request.form['password']) < 6:
            flash('Password must be at least 6 characters long')
            return redirect(url_for('add_user'))
        
        # Check if user already exists
        df = pd.read_csv('users.csv')
        if any(df['email'] == new_user['email']):
            flash('Email already exists')
            return redirect(url_for('add_user'))
            
        if any(df['roll_no'].astype(str) == str(new_user['roll_no'])):
            flash('Roll number already exists')
            return redirect(url_for('add_user'))
        
        # Add user to users.csv
        df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
        df.to_csv('users.csv', index=False)
        
        # Create directory for user's face data
        username = f"{new_user['first_name']}_{new_user['roll_no']}"
        userimagefolder = f"static/faces/{username}"
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        
        # Collect face data
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                flash('Failed to open camera. Please try again.')
                return redirect(url_for('admin_dashboard'))
            
            i = 0
            while i < 50:  # Collect 50 images
                ret, frame = cap.read()
                if not ret:
                    break
                    
                faces = extract_faces(frame)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    name = f'{username}_{i}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, name), face_img)
                    i += 1
                    
        except Exception as e:
            flash(f'Error capturing face data: {str(e)}')
            return redirect(url_for('admin_dashboard'))
            
        finally:
            if cap:
                cap.release()
        
        # Train the model with new face data
        if train_model():
            flash('User added successfully and face data captured')
        else:
            flash('User added but there was an error training the model')
            
        return redirect(url_for('admin_dashboard'))
        
    return render_template('add_user.html')

@app.route('/user/dashboard')
@login_required
def user_dashboard():
    df = pd.read_csv('users.csv')
    user_data = df[df['email'] == session['user']].iloc[0]
    
    # Calculate attendance statistics
    stats = {}
    user_roll = user_data['roll_no']
    
    # Get all attendance files from the last 30 days
    dates = [(datetime.now() - timedelta(days=i)).strftime("%m_%d_%y") for i in range(30)]
    attendance_data = []
    present_days = 0
    
    for date_str in dates:
        try:
            df = pd.read_csv(f'Attendance/Attendance-{date_str}.csv')
            if str(user_roll) in df['Roll'].astype(str).values:
                present_days += 1
                attendance_data.append(1)
            else:
                attendance_data.append(0)
        except FileNotFoundError:
            attendance_data.append(0)
    
    stats['total_days'] = len(dates)
    stats['present_days'] = present_days
    stats['attendance_percentage'] = (present_days / len(dates) * 100) if len(dates) > 0 else 0
    stats['dates'] = [datetime.strptime(d, "%m_%d_%y").strftime("%d-%b") for d in dates]
    stats['attendance_data'] = attendance_data
    
    return render_template('user_dashboard.html', user=user_data, stats=stats)

@app.route('/user/update', methods=['GET', 'POST'])
@login_required
def update_user():
    if request.method == 'POST':
        try:
            df = pd.read_csv('users.csv')
            user_idx = df[df['email'] == session['user']].index[0]
            
            # Update modifiable fields
            df.loc[user_idx, 'mobile'] = request.form['mobile']
            df.loc[user_idx, 'address'] = request.form['address']
            
            df.to_csv('users.csv', index=False)
            flash('Profile updated successfully')
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': True})
            return redirect(url_for('user_dashboard'))
            
        except Exception as e:
            flash(f'Error updating profile: {str(e)}')
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'message': str(e)})
            return redirect(url_for('update_user'))
        
    df = pd.read_csv('users.csv')
    user_data = df[df['email'] == session['user']].iloc[0]
    return render_template('update_user.html', user=user_data)

@app.route('/admin/manage_users')
@admin_required
def manage_users():
    df = pd.read_csv('users.csv')
    users = df[df['role'] != 'admin'].to_dict('records')  # Exclude admin users
    return render_template('manage_users.html', users=users)

@app.route('/admin/delete_user/<email>')
@admin_required
def delete_user(email):
    try:
        df = pd.read_csv('users.csv')
        user = df[df['email'] == email].iloc[0]
        username = f"{user['first_name']}_{user['roll_no']}"
        
        # Delete user's face data
        face_folder = f"static/faces/{username}"
        if os.path.exists(face_folder):
            for file in os.listdir(face_folder):
                os.remove(os.path.join(face_folder, file))
            os.rmdir(face_folder)
        
        # Remove user from users.csv
        df = df[df['email'] != email]
        df.to_csv('users.csv', index=False)
        
        # Retrain the model
        if train_model():
            flash(f'User {username} deleted successfully')
        else:
            flash('User deleted but model training failed')
            
    except Exception as e:
        flash(f'Error deleting user: {str(e)}')
    
    return redirect(url_for('manage_users'))

@app.route('/admin/daily_report')
@admin_required
def daily_report():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        total_users = len(pd.read_csv('users.csv')[pd.read_csv('users.csv')['role'] != 'admin'])
        present_count = len(df)
        absent_count = total_users - present_count
        
        attendance_data = {
            'date': datetoday2,
            'total_users': total_users,
            'present': present_count,
            'absent': absent_count,
            'attendance_percentage': (present_count/total_users * 100) if total_users > 0 else 0,
            'details': df.to_dict('records')
        }
        
        return render_template('daily_report.html', data=attendance_data)
    except Exception as e:
        flash(f'Error generating daily report: {str(e)}')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/weekly_report')
@admin_required
def weekly_report():
    try:
        # Get dates for the last 7 days
        dates = [(datetime.now() - timedelta(days=i)).strftime("%m_%d_%y") for i in range(7)]
        
        weekly_data = []
        for date_str in dates:
            try:
                df = pd.read_csv(f'Attendance/Attendance-{date_str}.csv')
                total_users = len(pd.read_csv('users.csv')[pd.read_csv('users.csv')['role'] != 'admin'])
                present_count = len(df)
                
                daily_data = {
                    'date': datetime.strptime(date_str, "%m_%d_%y").strftime("%d-%B-%Y"),
                    'present': present_count,
                    'absent': total_users - present_count,
                    'attendance_percentage': (present_count/total_users * 100) if total_users > 0 else 0
                }
                weekly_data.append(daily_data)
            except FileNotFoundError:
                # If no attendance file exists for that day
                weekly_data.append({
                    'date': datetime.strptime(date_str, "%m_%d_%y").strftime("%d-%B-%Y"),
                    'present': 0,
                    'absent': total_users,
                    'attendance_percentage': 0
                })
        
        return render_template('weekly_report.html', data=weekly_data)
    except Exception as e:
        flash(f'Error generating weekly report: {str(e)}')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/monthly_report')
@admin_required
def monthly_report():
    try:
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        # Get all days in current month
        num_days = calendar.monthrange(current_year, current_month)[1]
        dates = [(datetime(current_year, current_month, day)).strftime("%m_%d_%y") 
                for day in range(1, num_days + 1)]
        
        monthly_data = []
        total_users = len(pd.read_csv('users.csv')[pd.read_csv('users.csv')['role'] != 'admin'])
        
        for date_str in dates:
            try:
                df = pd.read_csv(f'Attendance/Attendance-{date_str}.csv')
                present_count = len(df)
                
                daily_data = {
                    'date': datetime.strptime(date_str, "%m_%d_%y").strftime("%d-%B-%Y"),
                    'present': present_count,
                    'absent': total_users - present_count,
                    'attendance_percentage': (present_count/total_users * 100) if total_users > 0 else 0
                }
                monthly_data.append(daily_data)
            except FileNotFoundError:
                monthly_data.append({
                    'date': datetime.strptime(date_str, "%m_%d_%y").strftime("%d-%B-%Y"),
                    'present': 0,
                    'absent': total_users,
                    'attendance_percentage': 0
                })
        
        return render_template('monthly_report.html', 
                             data=monthly_data, 
                             month=datetime.now().strftime("%B %Y"))
    except Exception as e:
        flash(f'Error generating monthly report: {str(e)}')
        return redirect(url_for('admin_dashboard'))

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
