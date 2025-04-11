// Import Firebase SDKs
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDl4w4TKuiSkhrL03u2oD1pYHLkT1F1NjE",
  authDomain: "authentication-29443.firebaseapp.com",
  projectId: "authentication-29443",
  storageBucket: "authentication-29443.firebasestorage.app",
  messagingSenderId: "760140940581",
  appId: "1:760140940581:web:562e1ced8bdc1843abc6cc",
  measurementId: "G-H6J2VX5Z46"
};

const app = initializeApp(firebaseConfig);

const auth = getAuth(app);

export { auth };
