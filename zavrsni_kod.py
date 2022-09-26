import cv2 as cv
import numpy as np
import dlib

title_window = 'Virtual Makeup'
secondary_window = 'Mode Changer'

cv.namedWindow(title_window) #pravljenje novog prozora

cap = cv.VideoCapture(0,cv.CAP_DSHOW)#pokrece se snimanje videa sa kamere



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/cavala/PycharmProjects/pythonProject/venv/Lib/site-packages"
                                 "/dlib-models-master/shape_predictor_68_face_landmarks.dat") #file koji sadrzi landmarkvoe lica(brojeve koji označuju dijelove lica

def nothing(x):
    pass

def on_trackbar(x): #funkcija za slidere koja ne radi nista
    pass

def createMask(frame,points,offset): #funkcija za pravljenje maske od predanih točaka

        mask = np.zeros_like(frame)
        mask = cv.fillPoly(mask,[points],(255,255,255),offset=offset)
        return mask

cv.createTrackbar('R', title_window, 0, 255, on_trackbar) #pravljenje 4 slidera
cv.createTrackbar('G', title_window, 0, 255, on_trackbar)
cv.createTrackbar('B', title_window, 0, 255, on_trackbar)
cv.createTrackbar('Alpha',title_window,0,100,on_trackbar)
switch = 'OFF|ON'
cv.createTrackbar(switch, title_window,0,1,on_trackbar)
mode = 'Mode'
cv.createTrackbar(mode, title_window,0,3,nothing) # slider za alpha vrijednosti




while (1):

    trac_mode = cv.getTrackbarPos(mode,title_window)
    ret, frame = cap.read()  # dekodira i vraca frame videa koji je u ovom slucaju web kamera
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # pretvara boje tog framea u sive
    originalFrame = frame.copy()
    faces = detector(frame)  # frame predajemo detectoru koji ce pornalazit lice na tom frameu
    r = cv.getTrackbarPos('R', title_window)  # postavljamo 4 slidera
    g = cv.getTrackbarPos('G', title_window)
    b = cv.getTrackbarPos('B', title_window)
    s = cv.getTrackbarPos(switch, title_window)
    a = cv.getTrackbarPos('Alpha',title_window)

    for face in faces:
        landmarks = predictor(gray_frame,
                              face)  # trazimo landmarke pomocu predictora te mu predajemo sivi frame i koordinate lica
        LandMarkPoints = []  # polje kojoj cemo predati koordinate svih landmarkova
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        LandMarkPoints.append([x, y])

    LandMarkPoints = np.array(LandMarkPoints)
    if(trac_mode == 0):

        lipsMask = createMask(frame, LandMarkPoints[48:68],None) #zovemo funkciju za pravljenje maske te predajemo landmarkove namjenjene za usne
        lipsColor = np.zeros_like(lipsMask) #vraca polje nula ali sa istim tipom i oblikom maske, samo su postavke maske vrijednosti nula

        if s==0: #ako je switch iskljucen boju maske postavlja na nule, mijenjanjem R,G,B slidera nece promjeniti boju usana
            lipsColor[:]=0
            a = 0
        else: #ako je switch upaljen mijenjat ce se boja usana
            lipsColor[:] = b, g, r

        a = a/50 #dijeljenje alpha vrijednosti slidera zbog vece preciznosti postavljenja alpha vrijednosti
        lipsColor = cv.bitwise_and(lipsMask, lipsColor) #funkcija koja radi konjukciju(I operator) polja lipsColor i polja lipsMask, tj spaja ih u jedno polje
        lipsColor = cv.GaussianBlur(lipsColor, (7, 7), 10) #dodaje se blur da rubovi maske nije ostri nega da izgleda malo glađe
        lipsColor = cv.addWeighted(originalFrame, 1, lipsColor, a, 0) # funkcija koja stvara takozvani "blend" alpha
        # vrijednosti poje(dinačnog framea znaci da ce originalni frame biti kompletno vidljiv a maska ce biti malo prozirna

        cv.imshow(title_window, lipsColor) #prikazujemo obojanu masku na prozoru

    elif(trac_mode==1):
        eyebrow_R_mask = createMask(frame, LandMarkPoints[17:22],None)
        eyebrow_R_Color = np.zeros_like(eyebrow_R_mask)

        eyebrow_L_mask = createMask(frame, LandMarkPoints[22:27],None)
        eyebrow_L_Color = np.zeros_like(eyebrow_L_mask)

        if s == 0:
            eyebrow_L_Color[:] = 0
            eyebrow_R_Color[:] = 0
        else:
            eyebrow_R_Color[:] = b, g, r
            eyebrow_L_Color[:] = b, g, r

        a = a / 50
        eyebrow_R_Color = cv.bitwise_and(eyebrow_R_mask, eyebrow_R_Color)
        eyebrow_L_Color = cv.bitwise_and(eyebrow_L_mask, eyebrow_L_Color)

        eyebrow_R_Color = cv.GaussianBlur(eyebrow_R_Color, (7, 7), 20)
        eyebrow_L_Color = cv.GaussianBlur(eyebrow_L_Color, (7, 7), 20)

        eyebrow_L_Color = cv.addWeighted(originalFrame, 1, eyebrow_L_Color, a, 0)
        eyebrow_R_Color = cv.addWeighted(originalFrame, 0, eyebrow_R_Color, a, 0)

        both_masks = cv.add(eyebrow_L_Color,eyebrow_R_Color)

        cv.imshow(title_window,both_masks)

    elif(trac_mode==2):
        offset_left_lash = [5,-5] #offset za makse
        offset_right_lash = [-5,-5]
        eyelash_L_Mask = createMask(frame, LandMarkPoints[42:46],offset_left_lash)
        eyelash_R_Mask = createMask(frame, LandMarkPoints[36:40],offset_right_lash)

        eyelash_L_Color = np.zeros_like(eyelash_L_Mask)
        eyelash_R_Color = np.zeros_like(eyelash_R_Mask)
        if s == 0:
            eyelash_L_Color[:] = 0
            eyelash_R_Color[:] = 0
        else:
            eyelash_L_Color[:] = b, g, r
            eyelash_R_Color[:] = b, g, r
        a = a/50
        eyelash_L_Color = cv.bitwise_and(eyelash_L_Mask,eyelash_L_Color)
        eyelash_R_Color = cv.bitwise_and(eyelash_R_Mask,eyelash_R_Color)

        eyelash_L_Color = cv.GaussianBlur(eyelash_L_Color, (7, 7), 20)
        eyelash_R_Color = cv.GaussianBlur(eyelash_R_Color, (7, 7), 20)

        eyelash_L_Color = cv.addWeighted(originalFrame, 1, eyelash_L_Color, a, 0)
        eyelash_R_Color = cv.addWeighted(originalFrame, 0, eyelash_R_Color, a, 0)

        both_masks_lashes = cv.add(eyelash_L_Color,eyelash_R_Color) #spoje se dvije maske u jednu radi prikaza obje maske odjednom

        cv.imshow(title_window,both_masks_lashes)

    elif(trac_mode==3):
        offset_left_lid = [5,-12] #offset za masku
        offset_right_lid = [-5,-12]

        eyelid_R_Mask = createMask(frame,LandMarkPoints[36:42],offset_right_lid)
        eyelid_L_Mask = createMask(frame,LandMarkPoints[42:48],offset_left_lid)

        eyelid_L_Color = np.zeros_like(eyelid_L_Mask)
        eyelid_R_Color = np.zeros_like(eyelid_R_Mask)

        if s == 0:
            eyelid_L_Color[:] = 0
            eyelid_R_Color[:] = 0
        else:
            eyelid_L_Color[:] = b, g, r
            eyelid_R_Color[:] = b, g, r
        a = a / 50
        eyelid_L_Color = cv.bitwise_and(eyelid_L_Mask, eyelid_L_Color)
        eyelid_R_Color = cv.bitwise_and(eyelid_R_Mask, eyelid_R_Color)

        eyelid_L_Color = cv.GaussianBlur(eyelid_L_Color, (7, 7), 20)
        eyelid_R_Color = cv.GaussianBlur(eyelid_R_Color, (7, 7), 20)

        eyelid_L_Color = cv.addWeighted(originalFrame, 1, eyelid_L_Color, a, 0)
        eyelid_R_Color = cv.addWeighted(originalFrame, 0, eyelid_R_Color, a, 0)

        both_masks_lids = cv.add(eyelid_L_Color, eyelid_R_Color) #spoje se dvije maske u jednu radi prikaza obje maske odjednom

        cv.imshow(title_window, both_masks_lids)



    if cv.waitKey(1) & 0xFF == ord('q'): # gasenje prozora pritiskom tipke "q"
        break

cv.destroyAllWindows() #funkcija unistava sav GUI koji se prikazuje radi memorije
