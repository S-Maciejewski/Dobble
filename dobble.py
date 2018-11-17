import matplotlib.gridspec as gridspec
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from random import randint
from sklearn.cluster import KMeans


def process_gamma(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((v / 255.0) ** inv_gamma) * 255 for v in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(cv2.LUT(img, table), table)


def linear_adjustment(x, minx, maxx, minvalue, maxvalue):
    return (maxx - x)/(maxx-minx) * (maxvalue - minvalue) + minvalue


def gamma_filter(img, aim):
    img_mean = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print("mean ", img_mean)
    gamma = 1
    if aim == "white_bg":
        gamma = linear_adjustment(img_mean, 150, 167, 1.1, 1.5)
    if aim == "card":
        gamma = linear_adjustment(img_mean, 153, 240, 0.6, 1.5)

    print(gamma)

    return process_gamma(img, gamma)


def coords(ccontour, offset):
    coord_xmin = int(np.amin(ccontour[:, 0, 1])) - offset
    coord_xmax = int(np.amax(ccontour[:, 0, 1])) + offset
    coord_ymin = int(np.amin(ccontour[:, 0, 0])) - offset
    coord_ymax = int(np.amax(ccontour[:, 0, 0])) + offset
    return coord_xmin, coord_xmax, coord_ymin, coord_ymax


def draw_arrow(img_arrows, p1, p2):
    cv2.arrowedLine(img_arrows, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                    (randint(0, 255), randint(0, 255), randint(0, 255)), 5)
    return img_arrows


# zapisywanie numeru symbolu na zdjęciu
def draw_number(img_arrows, number, coordinates):
    cv2.putText(img_arrows, number, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 153), 4)
    return img_arrows


def hsv_values(img):
    img_in_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return np.mean(img_in_hsv[:, :, 0]), np.mean(img_in_hsv[:, :, 1]), np.mean(img_in_hsv[:, :, 2])


def compare_hsv_values(pic1, pic2):
    hue1, sat1, val1 = hsv_values(pic1)
    hue2, sat2, val2 = hsv_values(pic2)
    diff = (hue1 - hue2)**2 + (sat1 - sat2)**2 + (val1 - val2)**2
    return diff


def compare_ratios(pic1, pic2):
    w1, h1 = pic1.shape[0:2]
    w2, h2 = pic2.shape[0:2]
    r1 = w1/h1 if w1 > h1 else h1/w1
    r2 = w2/h2 if w2 > h2 else h2/w2
    return abs(r1-r2)


def match_hu(img_arrows, card1, card2):

    great_matches = []
    for a, sign1 in enumerate(card1['signs']):
        for b, sign2 in enumerate(card2['signs']):
            # print(sign1["pic"].shape, sign2["pic"].shape)
            hu_moment = cv2.matchShapes(sign1["contour"], sign2["contour"], 1, 0.0)  # różnica w obiektach
            color_match = compare_hsv_values(sign1["pic"], sign2["pic"])
            ratio_match = compare_ratios(sign1["pic"], sign2["pic"])
            if hu_moment < 0.4 and ratio_match < 0.30:
                print(a, b, " : ", hu_moment, "    ", color_match, "     ", ratio_match)
                great_matches.append((sign1, sign2, a, b, hu_moment))

    if len(great_matches) != 0:
        best_match = great_matches[0]
        min_color_diff = 100000
        for match in great_matches:
            color_diff = compare_hsv_values(match[0]["pic"], match[1]["pic"]) + match[4]*1000
            if color_diff < min_color_diff:
                min_color_diff = color_diff
                best_match = match
                print(match[2], match[3])
            # print('symbols ', match[2], match[3], compare_hsv_values(match[0]["pic"], match[1]["pic"]))

        print("best match")
        p1 = best_match[0]['coords']
        p2 = best_match[1]['coords']
        img_arrows = draw_arrow(img_arrows, p1, p2)
        return img_arrows


def findMinRectangle(img, cnt, offset):
    rect = cv2.minAreaRect(cnt)
    rotated = cv2.warpAffine(img, cv2.getRotationMatrix2D(rect[0], rect[2], 1), img.shape[1::-1])
    cropped = rotated[int(rect[0][1]-rect[1][1]/2)-offset:int(rect[0][1]+rect[1][1]/2+offset),
                      int(rect[0][0]-rect[1][0]/2)-offset:int(rect[0][0]+rect[1][0]/2+offset)]
    xwidth, ywidth = cropped.shape[0:2]
    if xwidth < ywidth:
        cropped = np.rot90(cropped, 1)
    return cropped, rect[0]


def getRGBthresh(img, blue, green, red, counter):
    print("Card ", counter)
    ret1, img_blue_th = cv2.threshold(img[:, :, 0], blue, 255, cv2.THRESH_BINARY_INV)
    print("blue ", np.mean(img_blue_th))
    ret2, img_green_th = cv2.threshold(img[:, :, 1], green, 255, cv2.THRESH_BINARY_INV)
    print("green ", np.mean(img_green_th))
    ret3, img_red_th = cv2.threshold(img[:, :, 2], red, 255, cv2.THRESH_BINARY_INV)
    print("red ", np.mean(img_red_th))
    return cv2.bitwise_or(cv2.bitwise_or(img_blue_th, img_green_th, mask=None), img_red_th, mask=None)


def eraseBackground(img, contourslist, mode):
    stencil = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(stencil, contourslist, [255, 255, 255])
    stencil_inv = cv2.bitwise_not(stencil)
    result = cv2.bitwise_and(stencil, img)
    if mode == "white":
        result = cv2.add(result, stencil_inv)
    return result


def find_matches(file, allcards, number):
    img_col = cv2.imread(file)
    img_gray = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
    img_arrows = img_col
    cards = []

    if np.mean(img_gray) < 150:  # dla wszystkich zdjęć bez białego tła
        print("ciemne tło")
        # znalezienie konturów kart i wyczyszczenie tła dookoła
        ret, th1 = cv2.threshold(img_gray, 115, 255, cv2.THRESH_BINARY)
        th1 = cv2.erode(th1, np.ones((3, 3), np.uint8), iterations=3)
        th1 = cv2.dilate(th1, np.ones((3, 3), np.uint8), iterations=2)
        im2, contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # wycięcie kart
        for i, contour in enumerate(contours):
            cxmin, cxmax, cymin, cymax = coords(contour, 15)
            if cxmax-cxmin > 100 or cymax-cymin > 100:

                imgNoBg = eraseBackground(img_col, [contour], "white")
                card = imgNoBg[cxmin:cxmax, cymin:cymax]

                card = gamma_filter(card, "card")

                cv2.imwrite("./cards/" + str(i) + ".jpg", card)
                print("mean ", np.mean(cv2.cvtColor(card, cv2.COLOR_RGB2GRAY)))
                cardRGBthresh = getRGBthresh(card, 120, 110, 135, i)
                cardRGBthresh = cv2.dilate(cardRGBthresh, np.ones((3, 3), np.uint8), iterations=1)
                cv2.imwrite("./thresh/" + str(i) + ".jpg", cardRGBthresh)

                im2, contours, hierarchy = cv2.findContours(cardRGBthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # wycięcie znaków z kart
                signs = []
                for signContour in contours:
                    xmin, xmax, ymin, ymax = coords(signContour, 0)
                    if (xmax - xmin > 40 or ymax - ymin > 40) and not (xmin < 10 or ymin < 10 or xmax > card.shape[0] - 10):

                        cardNoBg = eraseBackground(card, [signContour], "white")
                        cardRGBthreshNoBg = eraseBackground(cardRGBthresh, [signContour], "black")

                        signPic, centerCoords = findMinRectangle(cardNoBg, signContour, 3)
                        signThPic, c = findMinRectangle(cardRGBthreshNoBg, signContour, 3)

                        coordx = centerCoords[1] + cxmin
                        coordy = centerCoords[0] + cymin

                        draw_number(img_arrows, str(len(signs)), (int(coordy), int(coordx)))

                        signs.append({"pic": signPic, "th": signThPic, "contour": signContour, "coords": [coordy, coordx]})

                cards.append({"pic": card, "signs": signs})
                allcards.append({"pic": card, "signs": signs})


    else:    # dla zdjęć z białym tłem
        print("białe tło")
        img_col = gamma_filter(img_col, "white_bg")
        print("mean after ", np.mean(cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)))
        img_th = getRGBthresh(img_col, 135, 135, 165, 1)
        # img_th = cv2.dilate(img_th, np.ones((3, 3), np.uint8), iterations=1)
        cv2.imwrite("./th1.jpg", img_th)
        im2, contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img_th = cv2.erode(img_th, np.ones((3, 3), np.uint8), iterations=1)
        cv2.imwrite("./debug/1" + str(number) + ".jpg", img_col)

        # znalezienie poprawnych znaków na zdjęciu
        signsCenters = []
        properContours = []
        for contour in contours:
            xmin, xmax, ymin, ymax = coords(contour, 0)
            if 50 < xmax-xmin < 300 and ymax-ymin > 50 and not (xmin < 200 and ymax > 1400):
                signsCenters.append([(xmax+xmin)/2, (ymax+ymin)/2])
                properContours.append(contour)

        img_bg = eraseBackground(img_col, properContours, "white")
        cv2.imwrite("./debug/2" + str(number) + ".jpg", img_bg)

        # pogrupowanie znaków w karty
        signsIdentity = KMeans(n_clusters=math.ceil(len(signsCenters)/8), random_state=0).fit(signsCenters).labels_
        for k in range(int(len(signsCenters)/8)):
            signsContours = [contour for j, contour in enumerate(properContours) if signsIdentity[j] == k]
            signsList = []
            for signContour in signsContours:
                signPic, centerCoords = findMinRectangle(img_col, signContour, 0)
                signThPic, c = findMinRectangle(img_th, signContour, 0)
                draw_number(img_arrows, str(len(signsList)), (int(centerCoords[0]), int(centerCoords[1])))
                signsList.append({"pic": signPic, "th": signThPic, "contour": signContour, "coords": centerCoords})
            cards.append({"pic": None, "signs": signsList})

            # TODO wycięcie karty, niekoniecznie potrzebne, ale fajne do debugu

    # calculate differences and pick best match
    for i in range(len(cards)):
        for j in range(i+1, len(cards)):
            print("Cards ", i, j)
            match_hu(img_arrows, cards[i], cards[j])

    cv2.imwrite("./img_arrows" + str(number) + ".jpg", img_arrows)

    for j, card in enumerate(cards):
        for i, sign in enumerate(card["signs"]):
            cv2.imwrite("./thresh/sign" + str(j) + str(i) + ".jpg", sign["th"])
            hsv = cv2.cvtColor(sign["pic"], cv2.COLOR_RGB2HSV)
            print("Card ", j, " sign ", i, "S: ", np.mean(hsv[:, :, 0]), "H: ", np.mean(hsv[:, :, 1]), "V: ", np.mean(hsv[:, :, 2]))

    for j, card in enumerate(cards):
        allcards.append(card)
    return allcards


allcards = []
# zdjęcia do poprawki
# 03 - karty za blisko krawędzi
# 07 - karty za blisko krawędzi
# 08 - za dużo duszków
# 09 - za dużo duszków
# 10 - nieznany błąd
# 15 - nieznany błąd
# 16 - mniej niebieskich znaków
# 17 - zerowe rozpoznanie znaków
# 18 - dużo niebieskich znaków
# 20 - karty za blisko siebie
# 21 - trochę za jasno
# 26 - karty za blisko krawędzi


# files = ["01", "02", "04", "05", "06", "08", "09", "11", "12", "13", "14", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "27"]
files = ["13", "14", "20", "27"]
for filenumber, file in enumerate(files):
    allcards = find_matches("./img/dobble"+ file +".jpg", allcards, filenumber)

for j, card in enumerate(allcards):
    if card["pic"] is not None:
        cv2.imwrite("./cards/card" + str(j) + ".jpg", card["pic"])

print(min, max)