from PIL import Image, ImageSequence

# 원본 GIF 열기
im = Image.open("input.gif")

# 자를 영역 지정 (left, upper, right, lower)
# 예: 위 30, 아래 40 픽셀 잘라내기
top_crop = 30
bottom_crop = 40
left, top, right, bottom = 0, top_crop, im.width, im.height - bottom_crop

frames = []
for frame in ImageSequence.Iterator(im):
    frame = frame.copy().crop((left, top, right, bottom))
    frames.append(frame)

# 잘라낸 GIF 저장
frames[0].save(
    "output_cropped.gif",
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=im.info["duration"],
    disposal=2
)