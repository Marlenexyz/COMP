import cv2
import time
import numpy as np


cap = cv2.VideoCapture('https://192.168.1.76:8080/video')


def compute_line(points):
    return np.cross(np.append(points[:2], 1), np.append(points[2:], 1))

def calculate_intersections(lines, shape_frame):
    intersects = []
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i != j:
                l1 = compute_line(lines[i])
                l2 = compute_line(lines[j])
                x12 = np.cross(l1, l2)
                if x12[2] == 0:
                    continue
                else:
                    x12 = x12 / x12[2]
                    if abs(x12[0]) < shape_frame[1] and abs(x12[1]) < shape_frame[0]:
                        if list(x12[:2]) not in intersects:
                            intersects.append(list(x12[:2]))
    return np.array([[int(j) for j in i] for i in intersects])



new_width, new_height = 300, 500
lower_threshold, upper_threshold = 250, 500
_maxLineGap, _minLineLength = 500, 250
window_width, window_height = 400, 300

target_rectangle = np.array([[0, 0],[window_width, 0],[window_width, window_height],[0, window_height]])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading from the camera.")
        break
    
    # resize frame
    # frame = cv2.resize(frame, (new_width, new_height))

    start = time.time()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5,5),0)
    # edges = cv2.Canny(gray, lower_threshold, upper_threshold)

    edges = cv2.Canny(frame, lower_threshold, upper_threshold)
    edge_points_yx = np.column_stack(np.where(edges > 0))
    edge_points_xy = edge_points_yx[:, [1, 0]]
    
    # Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, minLineLength=_minLineLength, maxLineGap=_maxLineGap)
    
    if lines is not None:
        lines = lines.reshape(-1, 4)
        selected_lines = lines[:4]
        for line in selected_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        intersections = calculate_intersections(selected_lines, frame.shape)
        for intersection in intersections:
            cv2.circle(frame, intersection, 5, (0, 255, 0), -1)
        
        ## Try Homography
        if len(intersections) == 4:                 ## only if all intersections
            try:
                rectangle = np.zeros((4, 2))

                min_sum_index = np.argmin(intersections[:, 0] + intersections[:, 1])     ## left-top and right-bottom
                max_sum_index = np.argmax(intersections[:, 0] + intersections[:, 1])
                rectangle[0] = intersections[min_sum_index]
                rectangle[2] = intersections[max_sum_index]

                diff = np.diff(intersections, axis=1)                                    ## right top and left-bottom
                min_diff_index = np.argmin(diff)
                max_diff_index = np.argmax(diff)
                rectangle[1] = intersections[min_diff_index]
                rectangle[3] = intersections[max_diff_index]
                homography, _ = cv2.findHomography(rectangle, target_rectangle)
                rectified_image = cv2.warpPerspective(frame, homography, (window_width, window_height))
                
                cv2.imshow('rectified', rectified_image)
            except:
                pass
   
    end = time.time()
    fps = 1 / (end-start)
    cv2.putText(frame, f'FPS: {round(fps, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("edges", edges)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()