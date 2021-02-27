''' [TEMP CODE] 0
gaze.refresh(frame)

gazeFrame = gaze.annotated_frame()
text = ""

if gaze.is_blinking():
    text = "깜빡임\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
elif gaze.is_right():
    text = "오른쪽 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
elif gaze.is_left():
    text = "왼쪽 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
elif gaze.is_center():
    text = "정면 응시\n\n왼쪽 눈동자 좌표: \n\n오른쪽 눈동자 좌표: "
else:
    text = "눈동자 인식 불가"

pill_image = Image.fromarray(cv2.cvtColor(gazeFrame, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pill_image)
draw.text((50, 50), text, font=ImageFont.truetype('NanumGothic-Bold.ttf', 30), fill=(0, 255, 0))
frame = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

left_pupil = gaze.pupil_left_coords()
right_pupil = gaze.pupil_right_coords()

cv2.putText(frame, str(left_pupil), (280, 135), cv2.FONT_HERSHEY_PLAIN, 2.2, (0, 255, 0), 2)
cv2.putText(frame, str(right_pupil), (305, 195), cv2.FONT_HERSHEY_PLAIN, 2.2, (0, 255, 0), 2)
'''

