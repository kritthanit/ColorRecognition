import cv2  # pip install opencv-python
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle


def cal_hist(img_loc, flag=False):
    if flag:
        img = img_loc
    else:
        img = cv2.imread(img_loc)
    mask = None
    hist_size = 256
    ranges = [0, 255]
    b_hist = cv2.calcHist([img], [0], mask, [hist_size], ranges)
    g_hist = cv2.calcHist([img], [1], mask, [hist_size], ranges)
    r_hist = cv2.calcHist([img], [2], mask, [hist_size], ranges)

    b_el = np.argmax(b_hist)
    g_el = np.argmax(g_hist)
    r_el = np.argmax(r_hist)

    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y_hist = cv2.calcHist([gry], [0], mask, [hist_size], ranges)
    y_el = np.argmax(y_hist)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [hist_size], ranges)
    s_hist = cv2.calcHist([hsv], [1], mask, [hist_size], ranges)
    v_hist = cv2.calcHist([hsv], [2], mask, [hist_size], ranges)

    h_el = np.argmax(h_hist)
    s_el = np.argmax(s_hist)
    v_el = np.argmax(v_hist)

    tmp = [b_el, g_el, r_el, y_el, h_el, s_el, v_el]

    return tmp


def create_train_data():
    x_features = []
    y_features = []
    for (lb, cname) in zip(label_color, color_name):
        all_img = os.listdir('dataset/{0}'.format(cname))
        for fx in all_img:
            # print('dataset/{0}/{1}'.format(cname, fn))
            ft = cal_hist('dataset/{0}/{1}'.format(cname, fx))
            x_features.append(ft)
            y_features.append(lb)
            x_features.append(ft)
            y_features.append(lb)

    return x_features, y_features


def train_knn():
    # create train data
    x_feature, y_feature = create_train_data()
    # print(x_feature)
    # print(y_feature)
    x_train, x_test, y_train, y_test = train_test_split(x_feature, y_feature, test_size=0.20, random_state=42,
                                                        stratify=y_feature)

    # train knn
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)

    # show accuracy
    ans = knn.predict(x_test)
    rpt = classification_report(y_test, ans)
    print(rpt)

    # best_score = 0
    # best_k = 0
    # for m in range(1, 10):
    #     knn = KNeighborsClassifier(n_neighbors=m)
    #     knn.fit(x_train, y_train)
    #     ans = knn.predict(x_test)
    #     acc = accuracy_score(y_test, ans)
    #     print('Acc: ', acc)
    #     if acc > best_score:
    #         best_score = acc
    #         best_k = m
    # print('Best Accuracy: ', best_score)
    # print('Best K: ', best_k)

    # find best-k
    param_grid = {'n_neighbors': np.arange(1, 10)}
    knn_cv = GridSearchCV(knn, param_grid, cv=5)
    knn_cv.fit(x_train, y_train)
    print(knn_cv.best_params_)
    print(knn_cv.best_score_)

    # Re-fit KNN
    k = knn_cv.best_params_['n_neighbors']
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    # save knn model
    fx = open('knn_color.pickle', 'wb')
    pickle.dump(knn, fx)
    fx.close()


def mouse_click(event, x, y, flags, param):
    global B, G, R, x_pos, y_pos, clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked = True
        x_pos = x
        y_pos = y
        B, G, R = pic[y, x]
        B = int(B)
        G = int(G)
        R = int(R)


# 0. Train KNN-Model
color_name = ['black', 'blue', 'gray', 'green', 'light-blue', 'orange', 'red', 'violet', 'white', 'yellow']
label_color = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
B = G = R = x_pos = y_pos = 0
clicked = False
# train_knn()

# 1. Load pre-train model
fn = open('knn_color.pickle', 'rb')
knn_model = pickle.load(fn)
fn.close()

# 2. Load test image
test_img = '003.png'
test_set = cal_hist(test_img)

# 3. Predict model
Result = knn_model.predict([test_set])

pic = cv2.imread(test_img)
scale = 1.5
f_color = (255, 255, 255)
f_thickness = 2
f_loc = (15, 45)

cv2.namedWindow('Color Recognition App')
cv2.setMouseCallback('Color Recognition App', mouse_click)

while True:
    cv2.imshow('Color Recognition App', pic)
    if clicked:
        ref_point = [(x_pos, y_pos), (x_pos + 40, y_pos + 40)]
        roi_img = pic[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]].copy()
        test_set = cal_hist(roi_img, flag=True)
        Result = knn_model.predict([test_set])

        # cv2.rectangle(image, startpoint, endpoint, color, thickness, -1 fills entire rectangle
        cv2.rectangle(pic, (10, 20), (750, 60), (B, G, R), -1)
        text = '{0} R={1} G={2} B={3}'.format(color_name[Result[0]], R, G, B)
        cv2.putText(pic, text, f_loc, cv2.FONT_HERSHEY_PLAIN, scale,
                    f_color, f_thickness, cv2.LINE_AA)
        if R + G + B >= 600:
            cv2.putText(pic, text, f_loc, cv2.FONT_HERSHEY_PLAIN, scale,
                        (0, 0, 0), f_thickness, cv2.LINE_AA)
        clicked = False

    # break with 'esc'
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

