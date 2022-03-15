from flask import Flask, request, render_template, make_response, redirect
import numpy as np
import cv2
from PIL import Image




classes_path = "yolo.names"
cfg_path ="yolov4-custom.cfg"
weights_path = "yolov4-custom_last.weights"


#Load network
classes = open(classes_path).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

np.random.seed(11)
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/output', methods=["GET", "POST"])
def output():

	if request.method == "POST":

		image = request.files["file"]
		if image:
			image = Image.open(image)
			image = np.array(image)
			(H, W) = image.shape[:2]
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)	#ímage preprocessing
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			boxes = []
			confidences = []
			classIDs = []
			for output in layerOutputs:	#array of 3
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > 0.5 :
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
			
			idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.3)
			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the image
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
					cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

			ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR)	)
			response = make_response(jpeg.tobytes())
			response.headers['Content-Type'] = 'image/png'
			return response	#wrong
			
			# out_file = os.path.join('/output', 'out.jpg')
			# cv2.imwrite(out_file, image)
			# return render_template("index.html", user_image = out_file)
		else:
			return render_template('index.html', msg='Chọn file upload')

	else:
    		return redirect('/')
	

if __name__ == '__main__':
	app.debug = True
	app.run(host='0.0.0.0', port=8088)