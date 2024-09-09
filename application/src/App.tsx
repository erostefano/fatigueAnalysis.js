import React, {useEffect, useRef, useState} from 'react';
import './App.css';
import Webcam from "react-webcam";
import * as blazeface from '@tensorflow-models/blazeface';
import '@tensorflow/tfjs';

function App() {
    const webcamRef = useRef<Webcam>(null); // Create a typed reference to the webcam
    const [model, setModel] = useState<blazeface.BlazeFaceModel>();

    // Load the BlazeFace model
    useEffect(() => {
        blazeface.load()
            .then(model => {
                setModel(model);
                console.log('Model loaded')
            })
            .catch(error => console.error(error))
    }, []);

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
        </div>
    );
}

export default App;
