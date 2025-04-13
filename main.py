# import cv2
# import os
# import pickle
# import face_recognition
# import cvzone
# import numpy as np  # ✅ Add this if missing

# # Try different camera indexes (0, 1, 2)
# for i in range(-1, 3):
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"Camera {i} opened successfully")
#         break  # Stop at the first working camera

# # If no camera was found, exit
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# cap.set(3, 498)  # Set width
# cap.set(4, 298)   # Set height


# # Ensure the correct path and file type
# imgBackground = cv2.imread('Resources/background.png')  # Check the extension


# # Verify if image is loaded
# if imgBackground is None:
#     print("Error: Could not load the background image. Check file path and format.")
#     exit()
    

# folderModePath ='Resources/Modes'
# modePathList = os.listdir(folderModePath)

# imgModeList = []
# for path in modePathList:
#     imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
    
# #print(len(imgModeList))

# #load the encoding file
# print("Loading Encode File ...")
# file = open('EncodeFile.p','rb')
# encodeListKnownWithIds = pickle.load(file)
# file.close()
# encodeListKnown,studentIds = encodeListKnownWithIds
# print("Encode File Loaded")



# while True:
#     success, img = cap.read()

#     if not success:  # Handle camera read error
#         print("Error: Failed to capture image.")
#         break

#     imgS= cv2.resize(img,(0,0),None,0.25,0.25)
#     imgS= cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

#     faceCurFrame = face_recognition.face_locations(imgS)
#     encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    
#     img_resized = cv2.resize(img, (498, 298))
#     imgBackground[150:150+298,35:35+498]=img_resized
#     target_size = (226,350)  # (Width, Height) → swap the numbers based on your background
#     imgModeList[2] = cv2.resize(imgModeList[2], target_size)
#     imgBackground[44:44+350,700:700+226]=imgModeList[2]

#     for encodeFace, faceLoc in zip(encodeCurFrame , faceCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#     #     print("matches",matches)
#     #     print("facesDis", faceDis)


#     matchIndex = np.argmin(faceDis)
#     # print("Match Index", matchIndex)

#     if matches[matchIndex]:
#        # print("Known Face Detected")
#        # print(studentIds[matchIndex])
#        y1, x2, y2, x1 = faceLoc
#        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
#        bbox = 55+x1, 162+y1, x2-x1, y2-y1 
#        imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)



#     for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
#      matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#     faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    
#     best_match_index = None
#     if len(faceDis) > 0:  # Ensure there are known faces to compare
#         best_match_index = faceDis.argmin()  # Get index of the best match
    
#     if best_match_index is not None and matches[best_match_index]:  # If a match is found
#         student_id = studentIds[best_match_index]  # Get the ID of the matched person
#         print(f"Face recognized: {student_id}")

#         # Display the recognized name near the face
#         top, right, bottom, left = faceLoc
#         top, right, bottom, left = top*4, right*4, bottom*4, left*4  # Resize back

#         cv2.rectangle(imgBackground, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(imgBackground, student_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)




#     #cv2.imshow("webcam", img)
#     cv2.imshow("Face Attendance", imgBackground)

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#         break

# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import os
# import pickle
# import face_recognition
# import numpy as np

# # Initialize camera
# cap = None
# for i in range(5):  # Try indexes from 0 to 4
#     temp_cap = cv2.VideoCapture(i)
#     if temp_cap.isOpened():
#         cap = temp_cap  # Store the working camera
#         print(f"Camera {i} opened successfully")
#         break
#     temp_cap.release()

# if cap is None or not cap.isOpened():
#     print("Error: Could not open any camera.")
#     exit()

# cap.set(3, 700)  # Set width  
# cap.set(4, 500)  # Set height

# # Load background image
# imgBackground = cv2.imread('Resources/background.png')
# if imgBackground is None:
#     print("Error: Could not load the background image. Check file path and format.")
#     exit()

# # Load mode images
# folderModePath = 'Resources/Modes'
# modePathList = os.listdir(folderModePath)
# imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# # 🔹 Load known images and their encodings
# image_folder = "images"  # Path to the images folder
# known_face_encodings = []
# known_face_ids = []
# print(known_face_ids)
# # List of image filenames and corresponding student IDs
# image_files = {
#     "1111.png": "1111",
#     "1113.png": "1113",
#     "1114.png": "1114",
#     "1112.png": "1112",
#     "1115.png": "1115"
# }

# for filename, student_id in image_files.items():
#     image_path = os.path.join(image_folder, filename)

#     # Load the image
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Could not load image {filename}")
#         continue

#     # Convert to RGB
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Get face encoding (assuming each image contains only one face)
#     encodings = face_recognition.face_encodings(img_rgb)
#     if len(encodings) > 0:
#         known_face_encodings.append(encodings[0])  # Take the first encoding
#         known_face_ids.append(student_id)
#     else:
#         print(f"Warning: No face detected in {filename}")

# print("Loaded Known Faces:", known_face_ids)

# # 🔹 Start real-time face detection
# while True:
#     success, img = cap.read()
#     if not success:
#         print("Error: Failed to capture image.")
#         break

#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     faceCurFrame = face_recognition.face_locations(imgS)
#     encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

#     for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
#         matches = face_recognition.compare_faces(known_face_encodings, encodeFace)
#         faceDis = face_recognition.face_distance(known_face_encodings, encodeFace)

#         best_match_index = -1
#         if any(matches):  # Ensure there's at least one match
#             best_match_index = np.argmin(faceDis)

#         print("Best Match Index:", best_match_index)

#         if best_match_index != -1 and matches[best_match_index]:
#             student_id = known_face_ids[best_match_index]  # ✅ Get matched student ID
#             print(f"Detected Face: {student_id}")
#         else:
#             student_id = "Unknown"

#         # Convert face location from 1/4th scale to full scale
#         y1, x2, y2, x1 = [v * 4 for v in faceLoc]

#         # Draw a rectangle around detected face
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, str(student_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     img_resized = cv2.resize(img, (500, 300))
#     imgBackground[150:150+300, 35:35+500] = img_resized

#     cv2.imshow("Face Attendance", imgBackground)

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#         break

# cap.release()
# cv2.destroyAllWindows()





import cv2
import os
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import webbrowser

# -------------------- Config --------------------
image_folder = "images"
excel_file = "/Users/anushkabhatnagar/Desktop/attendance.xlsx" 
entry_exit_buffer = 10  # seconds buffer between entry and exit

# -------------------- Load known images --------------------
known_face_encodings = []
known_face_ids = []  # list of (name, roll_no)

image_files = {
    "1111.png": "Anushka 2310990376",
    "1112.png": "Jiya 2310990619",
    "1113.png": "Jhanak 2310990618",
    "1114.png": "Ishita 2310990777",
    "1115.png": "Prerika 2410991589",
    # "1116.png": "Yogesh 2310991246",
    # "1117.png": "Sahil 2310992524",
    # "1118.png": "Tarun 2310992514",
    # "1119.png": "Raghav 2310992187",
    # "1120.png": "Sukrit 2310992537",
    # "1121.png": "Swapnil 2410992312",
    # "1122.png": "Samarth 2410992202",
    # "1123.png": "Prannay 2410992179",
    # "1124.png": "Saksham 2410992201"
}

print("⏳ Loading known images...")

for filename, info in image_files.items():
    path = os.path.join(image_folder, filename)
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Could not load {filename}")
        continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)

    if encodings:
        encoding = encodings[0]
        name, roll_no = info.split(" ", 1)
        roll_no = str(int(roll_no))  # Fix scientific notation
        known_face_encodings.append(encoding)
        known_face_ids.append((name, roll_no))
    else:
        print(f"⚠️ No face found in {filename}")

print("✅ Loaded known faces:", known_face_ids)

# -------------------- Attendance Log --------------------
reset_file = True  # Change to False if you want to keep old data

columns = ["Name", "Roll Number", "Date", "Entry Time", "Exit Time"]

if reset_file or not os.path.exists(excel_file):
    df = pd.DataFrame(columns=columns)
    df.to_excel(excel_file, index=False)



last_seen = {}

# -------------------- Attendance Function --------------------
# def log_attendance(name, roll_no, mode):
#     now = datetime.now()
#     date_str = now.strftime("%Y-%m-%d")
#     time_str = now.strftime("%H:%M:%S")
#     roll_no = str(int(roll_no))  # Ensure numeric format

#     df = pd.read_excel(excel_file, dtype={"Roll Number": str})

#     if mode == "entry":
#         # Entry not already logged for today
#         mask = (df["Name"] == name) & (df["Roll Number"] == roll_no) & (df["Date"] == date_str)
#         if not mask.any():
#             new_row = {
#                 "Name": name,
#                 "Roll Number": roll_no,
#                 "Date": date_str,
#                 "Entry Time": time_str,
#                 "Exit Time": ""
#             }
#             excel_file = "/Users/anushkabhatnagar/Desktop/attendance.xlsx"
#             df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#             last_seen[(name, roll_no)] = now

#     elif mode == "exit":
#         # Update the most recent entry with missing Exit Time
#         mask = (df["Name"] == name) & (df["Roll Number"] == roll_no) & \
#                (df["Date"] == date_str) & (df["Exit Time"] == "")
#         if mask.any():
#             idx = df[mask].index[-1]
#             df.at[idx, "Exit Time"] = time_str
#             if (name, roll_no) in last_seen:
#                 del last_seen[(name, roll_no)]

#     # df.to_excel(excel_file, index=False)

    
#     df.to_excel(excel_file, index=False)



def log_attendance(name, roll_no, mode):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    roll_no = str(int(roll_no))  # Ensure numeric format

    df = pd.read_excel(excel_file, dtype={"Roll Number": str})

    if mode == "entry":
        mask = (df["Name"] == name) & (df["Roll Number"] == roll_no) & (df["Date"] == date_str)
        if not mask.any():
            new_row = {
                "Name": name,
                "Roll Number": roll_no,
                "Date": date_str,
                "Entry Time": time_str,
                "Exit Time": ""
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            last_seen[(name, roll_no)] = now

    elif mode == "exit":
        mask = (df["Name"] == name) & (df["Roll Number"] == roll_no) & \
               (df["Date"] == date_str) & (df["Exit Time"] == "")
        if mask.any():
            idx = df[mask].index[-1]
            df.at[idx, "Exit Time"] = time_str
            if (name, roll_no) in last_seen:
                del last_seen[(name, roll_no)]

    df.to_excel(excel_file, index=False)

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("📸 Starting camera. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to capture frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name, roll_no = "Unknown", ""

        if True in matches:
            best_match_index = np.argmin(face_distances)
            name, roll_no = known_face_ids[best_match_index]

            now = datetime.now()

            if (name, roll_no) in last_seen:
                diff = (now - last_seen[(name, roll_no)]).total_seconds()
                if diff > entry_exit_buffer:
                    log_attendance(name, roll_no, "exit")
            else:
                log_attendance(name, roll_no, "entry")

        y1, x2, y2, x1 = [v * 4 for v in face_location]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {roll_no}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Open Excel file after execution
import webbrowser
webbrowser.open(os.path.abspath(excel_file))