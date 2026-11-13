import React from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/fraunces';
import '@fontsource-variable/figtree';
import './styles/tokens.css';
import './styles/app.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(<React.StrictMode><App /></React.StrictMode>);
