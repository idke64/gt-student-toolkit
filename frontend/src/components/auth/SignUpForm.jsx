import { useState, useRef } from "react";
import { signUp } from "../../firebase/auth";
import { useStore } from "@nanostores/react";

export default function SignUpForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [errorMsg, setErrorMsg] = useState("");

  const [loading, setLoading] = useState(false);

  const handleSignUp = async (e) => {
    e.preventDefault();
    setLoading(true);

    if (password !== confirmPassword) {
      setErrorMsg("Passwords don't match");
      return;
    }

    setErrorMsg("");

    try {
      await signUp(name, email, password);
      // window.location.href = "/chat";
    } catch (error) {
      setErrorMsg("Failed to sign up. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="space-y-4" onSubmit={handleSignUp}>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Username
        </label>
        <input
          type="name"
          className="text-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Email
        </label>
        <input
          type="email"
          className="text-input"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Password
        </label>
        <input
          type="password"
          className="text-input"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Confirm Password
        </label>
        <input
          type="password"
          className="text-input"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
        />
      </div>

      <button
        type="submit"
        className="w-full btn-primary py-2"
        disabled={loading}
      >
        Sign Up
      </button>
    </form>
  );
}
