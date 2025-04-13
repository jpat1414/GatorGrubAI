import React from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

function Header() {
  const navigate = useNavigate();

  const handleLogoClick = () => {
    navigate("/");       // Go to homepage
    window.location.reload(); // Refresh the page
  };

  return (
    <div className="header">
  <div className="logo-area" onClick={handleLogoClick}>
    <img src="/gatorgrub-logo.png" alt="Gator Grub Logo" className="logo-img" />
    <div className="logo-text-stack">
      <div className="logo-text">Gator</div>
      <div className="logo-text">Grub</div>
    </div>
  </div>
</div>

  );
}

export default Header;
