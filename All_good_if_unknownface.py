import face_recognition, cv2, os, threading, random, time
import numpy as np
#import sounddevice as sd
#from scipy.io.wavfile import write

# Get a reference to webcam #0 (the default one)
#video_capture = cv2.VideoCapture("http://192.168.50.106:8910/video")
video_capture = cv2.VideoCapture(0)

#empty lists for encodings and names
known_face_encodings = []
known_face_names = []

#looping through known faces to use
for filename in os.listdir("known_faces"):
    os.chdir("known_faces")
    img = face_recognition.load_image_file(filename)
    known_face_encodings.append(face_recognition.face_encodings(img)[0])
    name = filename[:filename.index(".")]
    known_face_names.append(name)
    os.chdir("..")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

def video_feed():
    process_this_frame = True
    

    while True:
        #list of names of random images
        unknown_face_names = []
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                if name == "Unknown":
                    try:
                        os.chdir("known_faces")
                    except:
                        pass
                    while True:
                        unknown_name = str(random.randint(0,100000))
                        if unknown_name in unknown_face_names:
                            continue
                        else:
                            break
                    cv2.imwrite(filename=f"{unknown_name}.jpg", img=frame)
                    try:
                        os.chdir("known_faces")
                    except:
                        pass
                    image = cv2.imread(f"{unknown_name}.jpg")
                    t, r, b, l = face_locations[-1][0] *4, face_locations[-1][1] *4, face_locations[-1][2]*4, face_locations[-1][3]*4
                    crop = image[t-10:b+10, l-10:r+10]
                    try:
                        os.chdir("known_faces")
                    except:
                        pass
                    cv2.imwrite(filename=f"{unknown_name}.jpg", img=crop)
                    os.chdir("..")
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(unknown_name)
                    name = unknown_name

                    #fs = 44100  # Sample rate
                    #seconds = 10  # Duration of recording

                    #myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                    #sd.wait()  # Wait until recording is finished
                    #write('output.wav', fs, myrecording)  # Save as WAV file

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.namedWindow("Video")
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return None

def get_name():
    not_interupted = True
    while not_interupted:
        try:
            os.chdir("face_stream")
        except:
            pass
        for filename in os.listdir("known_faces"):
            k = cv2.waitKey(1) & 0xFF
            try:
                os.chdir("known_faces")
            except:
                pass
            try:
                int(filename[:filename.index(".")])
                cv2.namedWindow(filename[:filename.index(".")])
                image = cv2.imread(filename)
                cv2.imshow(filename[:filename.index(".")], image)
                actual_name = input("Who is this?:")
                encode_image = face_recognition.load_image_file(filename)
                known_face_encodings[known_face_names.index(filename[:filename.index(".")])] = face_recognition.face_encodings(encode_image)[0]
                known_face_names[known_face_names.index(filename[:filename.index(".")])] = actual_name
                os.rename(filename, f"{actual_name}.jpg") 
            except:
                pass
            os.chdir("..")
            if k == ord("q"):
                not_interupted = False
        time.sleep(0.02)

threading.Thread(target=video_feed, args=tuple()).start()
threading.Thread(target=get_name, args=tuple()).start()