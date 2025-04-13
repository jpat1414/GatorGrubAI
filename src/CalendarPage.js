import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Header from "./Header"; // ✅ Import the Header
import "./App.css";

function CalendarPage() {
  const [events, setEvents] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [matchedEvents, setMatchedEvents] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetch("/data.json")
      .then((res) => res.json())
      .then((data) => setEvents(data))
      .catch((err) => console.error("Failed to load data:", err));
  }, []);

  const handleDateChange = (e) => {
    const selected = e.target.value;
    setSelectedDate(selected);
    const matches = events.filter((ev) => ev.date === selected);
    setMatchedEvents(matches);
  };

  return (
    <>
      <Header /> {/* ✅ Add this at the top */}
      <div className="calendar-container">
        <h1 className="title">Club Events Calendar</h1>
        <input
          type="date"
          value={selectedDate}
          onChange={handleDateChange}
          className="date-picker"
        />
        <div className="event-results">
          {selectedDate && matchedEvents.length > 0 ? (
            matchedEvents.map((event, index) => (
              <div className="event-details" key={index}>
                <h2>{event.clubName}</h2>
                <p>
                  Source:{" "}
                  <a
                    href={event.sourceLink}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {event.sourceLink}
                  </a>
                </p>
              </div>
            ))
          ) : selectedDate ? (
            <p className="no-events">No events for this day.</p>
          ) : null}
        </div>
        <button className="back-button" onClick={() => navigate("/")}>
          ⬅ Back to Home
        </button>
      </div>
    </>
  );
}

export default CalendarPage;
