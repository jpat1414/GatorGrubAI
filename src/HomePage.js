import React from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <h1 className="home-title">Gator Grub</h1>
      <p className="home-subtitle">Description/about</p>
      <button className="home-button" onClick={() => navigate("/calendar")}>
        Get Started Now
      </button>
    </div>
  );
}

export default HomePage;
