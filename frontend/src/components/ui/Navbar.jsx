import { useState, useEffect } from "react";
import { useStore } from "@nanostores/react";
import { user, isAuth, logout } from "../../firebase/auth";

export default function Navbar() {
  const [currPath, setCurrPath] = useState("");
  const $isAuth = useStore(isAuth);
  const $user = useStore(user);

  const links = [
    { name: "Home", path: "/", protected: false },
    { name: "Chat", path: "/chat", protected: true },
    { name: "Resources", path: "/resources", protected: false },
  ];

  useEffect(() => {
    setCurrPath(window.location.pathname);
  }, []);

  const handleLogout = async () => {
    try {
      await logout();
      window.location.href = "/";
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  return (
    <nav className="bg-white shadow-md px-8 fixed w-full">
      <div className="flex justify-between h-16 relative">
        <div className="flex items-center gap-x-8">
          <a
            href="/"
            className="text-xl font-bold text-gray-900 hover:bg-neutral-100 rounded py-1 px-2"
          >
            <span className="text-primary-200">GT</span> Student Toolkit
          </a>
        </div>
        <div className="flex items-center gap-x-2 absolute top-1/2 -translate-y-1/2 left-1/2 -translate-x-1/2">
          {links.map(
            (link) =>
              ((link.protected && isAuth) || !link.protected) && (
                <a
                  key={link.path}
                  href={link.path}
                  className={`nav-link ${
                    currPath === link.path ? "after:w-[60%] bg-neutral-100" : ""
                  }`}
                >
                  {link.name}
                </a>
              )
          )}
        </div>
        <div className="flex items-center space-x-6">
          <div className="flex items-center gap-x-3">
            {$isAuth ? (
              <>
                <button className="nav-link">{$user?.email}</button>
                <button onClick={handleLogout} className="btn-red h-8 px-3">
                  Logout
                </button>
              </>
            ) : (
              <>
                <a href="/login" className="btn-primary h-8 px-3">
                  Login
                </a>
                <a href="/signup" className="btn-secondary h-8 px-3">
                  Sign up
                </a>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
