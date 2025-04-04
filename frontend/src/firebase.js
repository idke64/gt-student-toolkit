// Import Firebase SDKs
import { initializeApp } from "firebase/app";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword } from "firebase/auth";
import { getFirestore, collection, addDoc, setDoc, doc } from "firebase/firestore";

// Your Firebase configuration (make sure to replace with your actual config values)
const firebaseConfig = {
    apiKey: "AIzaSyDl4w4TKuiSkhrL03u2oD1pYHLkT1F1NjE",
    authDomain: "authentication-29443.firebaseapp.com",
    projectId: "authentication-29443",
    storageBucket: "authentication-29443.firebasestorage.app",
    messagingSenderId: "760140940581",
    appId: "1:760140940581:web:562e1ced8bdc1843abc6cc",
    measurementId: "G-H6J2VX5Z46"
  };

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Get instances of Firebase Authentication and Firestore
const auth = getAuth(app);
const db = getFirestore(app);

// Function to sign up a new user
const signUp = async (email, password) => {
  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;
    
    // Store user in Firestore
    const userRef = doc(db, "users", user.uid);
    await setDoc(userRef, {
      email: email,
      uid: user.uid,
      createdAt: new Date(),
    });

    return user;
  } catch (error) {
    throw new Error(error.message);
  }
};

// Function to log in a user
const login = async (email, password) => {
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    return userCredential.user;
  } catch (error) {
    throw new Error(error.message);
  }
};

export { auth, db, signUp, login };
