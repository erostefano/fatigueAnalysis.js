import React, {useEffect, useRef, useState} from 'react';
import './App.css';
import Webcam from 'react-webcam';
import * as blazeface from '@tensorflow-models/blazeface';
import '@tensorflow/tfjs';

function App() {
    const webcamRef = useRef<Webcam>(null); // Create a typed reference to the webcam
    const [model, setModel] = useState<blazeface.BlazeFaceModel>();
    const [faceImage, setFaceImage] = useState<string | null>(null);

    // Load the BlazeFace model
    useEffect(() => {
        blazeface.load()
            .then(model => {
                setModel(model);
                console.log('Model loaded');
            })
            .catch(error => console.error(error));
    }, []);

    // Function to capture and process image
    const captureAndProcess = async () => {
        const imageSrc = webcamRef.current?.getScreenshot();
        if (!imageSrc || !model) return;

        // Create an image element
        const image = new Image();
        image.src = imageSrc;
        await new Promise((resolve) => {
            image.onload = () => resolve(null);
        });

        // Create canvas for image processing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Set canvas dimensions
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);

        // Detect faces
        const predictions = await model.estimateFaces(canvas, false);

        if (predictions.length > 0) {
            // Process the first detected face
            const {topLeft, bottomRight} = predictions[0];
            const [x1, y1] = topLeft as [number, number];
            const [x2, y2] = bottomRight as [number, number];

            const faceWidth = x2 - x1;
            const faceHeight = y2 - y1;

            // Draw face on a new canvas
            const faceCanvas = document.createElement('canvas');
            faceCanvas.width = faceWidth;
            faceCanvas.height = faceHeight;
            const faceCtx = faceCanvas.getContext('2d');
            if (faceCtx) {
                faceCtx.drawImage(canvas, x1, y1, faceWidth, faceHeight, 0, 0, faceWidth, faceHeight);

                // Convert face canvas to image and update state
                const faceImg = faceCanvas.toDataURL('image/png');
                setFaceImage(faceImg);
            }
        }
    };

    return (
        <div className="App">
            <h2>Webcam Capture</h2>
            <Webcam
                audio={false} // Disable audio
                ref={webcamRef} // Attach the reference
                screenshotFormat="image/png" // Screenshot format
                width={320} // Set width
                height={240} // Set height
            />
            <button onClick={captureAndProcess}>Capture and Extract Face</button>
            {faceImage && (
                <div>
                    <h3>Detected Face</h3>
                    <img src={faceImage} alt="Detected Face"/>
                </div>
            )}
        </div>
    );
}

export default App;
