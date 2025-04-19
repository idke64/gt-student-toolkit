import { atom, computed } from "nanostores";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  updateProfile,
} from "firebase/auth";
import { auth } from "./firebase";

export const user = atom(null);

export const isAuth = computed(user, (user) => !!user);

onAuthStateChanged(auth, (currUser) => {
  user.set(
    currUser
      ? { email: currUser.email, id: currUser.uid, name: currUser.displayName }
      : null
  );
});

export async function login(email, password) {
  try {
    const userCredential = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );
    return userCredential.user;
  } catch (error) {
    throw new Error(error.message);
  }
}

export async function signUp(name, email, password) {
  try {
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      email,
      password
    );

    await updateProfile(userCredential.user, {
      displayName: name,
    });

    const firebaseUser = userCredential.user;
    const API_URL = import.meta.env.PUBLIC_API_URL;

    const response = await fetch(`${API_URL}/api/users`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        userId: firebaseUser.uid,
        name: name,
        email: firebaseUser.email,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Error creating user in backend:", errorData);
      throw new Error("Failed to create user in database");
    }

    return firebaseUser;
  } catch (error) {
    throw new Error(error.message);
  }
}

export async function logout() {
  try {
    await signOut(auth);
  } catch (error) {
    throw new Error(error.message);
  }
}
