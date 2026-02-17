document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form.form");
  const startDate = document.getElementById("start_date");
  const endDate = document.getElementById("end_date");
  const startTime = document.getElementById("start_time");
  const endTime = document.getElementById("end_time");
  const datetimeRangeError = document.getElementById("datetime-range-error");

  const toggleAll = document.getElementById("toggle-all");
  const predictorInputs = document.querySelectorAll(
    ".checkbox-grid input[type=\"checkbox\"]"
  );

  const latitude = document.getElementById("latitude");
  const longitude = document.getElementById("longitude");
  const latitudeError = document.getElementById("latitude-error");
  const longitudeError = document.getElementById("longitude-error");

  const north = document.getElementById("north");
  const west = document.getElementById("west");
  const south = document.getElementById("south");
  const east = document.getElementById("east");

  const northError = document.getElementById("north-error");
  const southError = document.getElementById("south-error");
  const westError = document.getElementById("west-error");
  const eastError = document.getElementById("east-error");

  const latitudeRangeError = document.getElementById("latitude-range-error");
  const longitudeRangeError = document.getElementById("longitude-range-error");
  const submitButton = document.getElementById("submit-button");

  if (toggleAll && predictorInputs.length) {
    toggleAll.addEventListener("click", () => {
      const allChecked = Array.from(predictorInputs).every(
        (input) => input.checked
      );
      predictorInputs.forEach((input) => {
        input.checked = !allChecked;
      });
      toggleAll.textContent = allChecked ? "Select All" : "Clear All";
    });
  }

  function maskDate(input) {
    if (!input) return;
    input.addEventListener("input", () => {
      let v = input.value.replace(/\D/g, "");
      if (v.length > 8) v = v.slice(0, 8);

      let year = v.slice(0, 4);
      let month = v.slice(4, 6);
      let day = v.slice(6, 8);

      const now = new Date();
      const currentYear = now.getFullYear();
      const previousMonth = now.getMonth();

      let y = null;
      if (year.length === 4) {
        y = parseInt(year, 10);
        if (y < 1940) year = "1940";
        if (y > 2026) year = "2026";
        y = parseInt(year, 10);
      }

      if (month.length === 2) {
        let m = parseInt(month, 10);
        if (m < 1) month = "01";
        if (m > 12) month = "12";
        if (y === currentYear && m > previousMonth) {
          month = previousMonth.toString().padStart(2, "0");
        }
      }

      if (day.length === 2) {
        let d = parseInt(day, 10);
        if (d < 1) day = "01";
        if (d > 31) day = "31";
      }

      let out = "";
      if (year.length) out = year;
      if (month.length) out += "-" + month;
      if (day.length) out += "-" + day;

      input.value = out;
    });
  }

  function maskTime(input) {
    if (!input) return;
    input.addEventListener("input", () => {
      let v = input.value.replace(/\D/g, "");
      if (v.length > 4) v = v.slice(0, 4);

      let hour = v.slice(0, 2);
      let minute = v.slice(2, 4);

      if (hour.length === 2) {
        let h = parseInt(hour, 10);
        if (h < 1) hour = "00";
        if (h > 23) hour = "23";
      }

      if (minute.length === 2) {
        let m = parseInt(minute, 10);
        if (m < 0) minute = "00";
        if (m > 59) minute = "59";
      }

      let out = "";
      if (hour.length) out = hour;
      if (minute.length) out += ":" + minute;

      input.value = out;
    });
  }

  function maskNonDigit(input) {
    if (!input) return;
    input.addEventListener("keypress", (e) => {
      if (!/[0-9]/.test(e.key)) e.preventDefault();
    });
  }

  function normalizeCoordinate(input) {
    if (!input) return;
    let v = input.value.replace(/[^\d.\-]/g, "");
    v = v.replace(/(?!^)-/g, "");
    const parts = v.split(".");
    if (parts.length > 2) {
      v = parts.shift() + "." + parts.join("");
    }
    input.value = v;
  }

  function setError(errorEl, message) {
    if (!errorEl) return;
    errorEl.textContent = message;
    if (message) {
      errorEl.classList.add("is-visible");
    } else {
      errorEl.classList.remove("is-visible");
    }
  }

  function validateCoordinate(input, errorEl, min, max) {
    if (!input) return true;
    const raw = input.value.trim();
    if (!raw) {
      setError(errorEl, "");
      input.removeAttribute("aria-invalid");
      return true;
    }

    const value = Number(raw);
    if (Number.isNaN(value) || value < min || value > max) {
      setError(errorEl, `Value must be between ${min} and ${max}.`);
      input.setAttribute("aria-invalid", "true");
      return false;
    }

    setError(errorEl, "");
    input.removeAttribute("aria-invalid");
    return true;
  }

  function validateCoordinateOrder(maxInput, minInput, errorEl, maxLabel, minLabel) {
    if (!maxInput || !minInput) return true;
    const maxRaw = maxInput.value.trim();
    const minRaw = minInput.value.trim();
    if (!maxRaw || !minRaw) {
      setError(errorEl, "");
      return true;
    }

    const maxVal = Number(maxRaw);
    const minVal = Number(minRaw);
    if (Number.isNaN(maxVal) || Number.isNaN(minVal)) {
      setError(errorEl, "");
      return true;
    }

    if (maxVal < minVal) {
      setError(errorEl, `${maxLabel} must be greater than or equal to ${minLabel}.`);
      return false;
    }

    setError(errorEl, "");
    return true;
  }

  function parseDateTime(dateValue, timeValue) {
    const datePattern = /^\d{4}-\d{2}-\d{2}$/;
    const timePattern = /^\d{2}:\d{2}$/;
    if (!datePattern.test(dateValue) || !timePattern.test(timeValue)) {
      return null;
    }

    const parsed = new Date(`${dateValue}T${timeValue}:00`);
    if (Number.isNaN(parsed.getTime())) {
      return null;
    }
    return parsed;
  }

  function validateDateTimeOrder() {
    if (!startDate || !endDate || !startTime || !endTime) return true;

    const startValue = parseDateTime(startDate.value.trim(), startTime.value.trim());
    const endValue = parseDateTime(endDate.value.trim(), endTime.value.trim());

    if (!startValue || !endValue) {
      setError(datetimeRangeError, "");
      return true;
    }

    if (endValue < startValue) {
      setError(datetimeRangeError, "End date/time must be after or equal to start date/time.");
      return false;
    }

    setError(datetimeRangeError, "");
    return true;
  }

  function updateFormValidity() {
    const latOk = validateCoordinate(latitude, latitudeError, -90, 90);
    const lonOk = validateCoordinate(longitude, longitudeError, -180, 180);
    const northOk = validateCoordinate(north, northError, -90, 90);
    const southOk = validateCoordinate(south, southError, -90, 90);
    const westOk = validateCoordinate(west, westError, -180, 180);
    const eastOk = validateCoordinate(east, eastError, -180, 180);
    const latRangeOk = validateCoordinateOrder(north, south, latitudeRangeError, "North", "South");
    const lonRangeOk = validateCoordinateOrder(east, west, longitudeRangeError, "East", "West");
    const datetimeOk = validateDateTimeOrder();

    const allOk = latOk && lonOk && northOk && southOk &&
      westOk && eastOk && latRangeOk && lonRangeOk && datetimeOk;

    if (submitButton) submitButton.disabled = !allOk;
    return allOk;
  }

  maskNonDigit(startDate);
  maskNonDigit(endDate);
  maskNonDigit(startTime);
  maskNonDigit(endTime);

  maskDate(startDate);
  maskDate(endDate);

  maskTime(startTime);
  maskTime(endTime);

  const coordinateInputs = [latitude, longitude, north, south, west, east].filter(
    Boolean
  );

  const datetimeInputs = [startDate, endDate, startTime, endTime].filter(Boolean);

  updateFormValidity();

  if (coordinateInputs.length) {
    coordinateInputs.forEach((input) => {
      input.addEventListener("input", () => {
        normalizeCoordinate(input);
        updateFormValidity();
      });
    });
  }

  if (datetimeInputs.length) {
    datetimeInputs.forEach((input) => {
      input.addEventListener("input", updateFormValidity);
      input.addEventListener("change", updateFormValidity);
    });
  }

  if (form) {
    form.addEventListener("submit", (event) => {
      if (!updateFormValidity()) {
        event.preventDefault();
      }
    });
  }
});
