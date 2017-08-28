# Script to filter out frames from a recording. Useful for recovery driving
# and only keep the recovery part, where the car comes back to the center, but
# don't keep the part when car goes out of center. Press q to quit and and
# press space to toggle keeping frames.
import csv
import cv2
import os
import sys
import time


def read_lines_from_driving_log(path):
    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    # First line contains the name of the columns for me. If it just for some
    # library versions, then it is anyway ok to skip first frame.
    lines = lines[1:]

    return lines


# Convert a set of lines from a driving log to datas and expected values.
def line_to_image_and_angle(line, indrivedir):
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = os.path.join(indrivedir, 'IMG', filename)
    image = cv2.imread(current_path)
    angle = float(line[3])

    return image,angle


def visual_manual_filtering(indrivelogpath, outdrivelogpath, timeperframe,
        framescale):
    indrivedir = os.path.dirname(indrivelogpath)
    lines = read_lines_from_driving_log(indrivelogpath)
    # True if key was down last frame.
    # Problem is that waitKey with time out does not wait for timeout when a
    # key is pressed. Therefore this cheep and ugly homemade toggle logic.
    waskeydown = False
    savingframes = False
    try:
        with open(outdrivelogpath, 'w') as outdrivelogfile:
            outdrivelog = csv.writer(outdrivelogfile, delimiter = ',')
            for line in lines:
                image,angle = line_to_image_and_angle(line, indrivedir)
                image = cv2.resize(image, None, fx=framescale,fy=framescale)
                cv2.putText(image, 'Angle: {:.2f}'.format(angle),
                        (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Savingframes: {}'.format(savingframes),
                    (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('display', image)
                ret = cv2.waitKey(timeperframe)
                if ret > 0:
                    ret = ret & 255
                    if ret == ord('q'):
                        # Press q to quit
                        return
                    elif not waskeydown and ret == ord(' '):
                        savingframes = not savingframes
                    waskeydown = True
                else:
                    waskeydown = False

                if savingframes:
                    outdrivelog.writerow(line)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    indrivelogpath, outdrivelogpath = sys.argv[1:3]
    timeperframe = int(sys.argv[3]) if len(sys.argv)>3 else 67
    framescale = float(sys.argv[4]) if len(sys.argv)>4 else 1.0
    visual_manual_filtering(indrivelogpath, outdrivelogpath, timeperframe,
            framescale)
