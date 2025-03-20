function clickButton() {
  // Use "#" to target the ID "button"
  const button = document.querySelector('#star-icon'); 

  if (button) {
    button.click();
    console.log('Button clicked using ID!');
  } else {
    console.log('Button not found.');
  }
}

setInterval(clickButton, 60000); // Click every 60 seconds
