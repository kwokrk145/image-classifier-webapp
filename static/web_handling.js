// Access the buttons and input elements
const input = document.getElementById('file-input');
const fileNameDiv = document.getElementById('file-name');

// Add an event listener to update the file name when a file is selected
input.addEventListener('change', () => {
  // iput.files is a FileList object, which contains the selected files
  if (input.files.length > 0) {
    fileNameDiv.textContent = input.files[0].name;
  } else {
    fileNameDiv.textContent = '';
  }
});