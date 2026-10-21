import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import "./favicon.ico";
import * as serviceWorkerRegistration from "./serviceWorkerRegistration";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// 🆕 Регистрация Service Worker для PWA
// unregister() → register() только когда готов включить офлайн-режим
serviceWorkerRegistration.register();