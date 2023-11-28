# Use OpenCV and Python to construct and demonstrate real-time detection and localization of a prominent line in live video.

import cv2
import numpy as np
import time

np.random.seed(1234)

cap = cv2.VideoCapture(1)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start = time.time()

    t_lower = 100                                     
    t_upper = 200                                      

    # edges = cv2.Canny(frame,t_lower,t_upper)
    edges = cv2.Canny(gray,t_lower,t_upper)         #lower and upper threshold

    edge_points = np.column_stack(np.where(edges > 0))              ## find non zero elements
    frame_edges = frame.copy()

    ##RANSAC
    # required_points = 2
    best_line = None
    max_inliers = 0
    num_iterations = 1000
    threshold = 2.0


    ## Reduced Number of Edge Points
    # k = 2
    # edge_points = edge_points[::k, :] 

    for i in range(num_iterations):
        sample = edge_points[np.random.choice(len(edge_points), size=2, replace=False)]
        x1, y1 = sample[0]
        x2, y2 = sample[1]

        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1


            # Calculate the Euclidean distance from each k-th point to the line
            distances = np.abs((m * edge_points[:, 0] - edge_points[:, 1] + b) / np.sqrt(m**2 + 1))     ##A * x + B * y + C / sqrt (A^2 + B ^2) 

            inliers = np.sum(distances < threshold)         ## counts inliers

            if inliers > max_inliers:                       ## best params
                max_inliers = inliers
                best_line = (m, b)
    frame_line = frame.copy()
    if best_line is not None:
        m, b = best_line
        y1 = 0
        x1 = int((y1 - b) / m)
        y2 = frame.shape[0]
        x2 = int((y2 - b) / m)
        cv2.line(frame_line, (y1, x1), (y2, x2), (0, 255, 0), 2)
        cv2.circle(frame_line, (y1, x1), 20, (0, 0, 255), -1)    ## Draw red circles at random edge points
        cv2.circle(frame_line, (y2, x2), 20, (0, 0, 255), -1)    ## Draw red circles at random edge points

    for x, y in edge_points:
        cv2.circle(frame_edges, (y, x), 1, (0, 255, 0), -1)    ## Draw green circles at edge points
    
    end = time.time()

    fps = 1 / (end - start)
    cv2.putText(frame_edges, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame_line, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('RANSAC', frame_line)
    cv2.imshow('edge', frame_edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if cv2.waitKey(1) & 0xFF == ord('w'):
    #     out = cv2.imwrite('capture.jpg',frame)
    #     break

cap.release()
cv2.destroyAllWindows()
