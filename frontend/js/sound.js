const audio = new Audio('audio/background.mp3');
const buttonSound = new Audio('audio/sound.mp3');

audio.loop = true;

let isMusicPlaying = true;
let isSoundEnabled = false;

const musicIcon = document.querySelector('.header__music-img');
const soundIcon = document.querySelector('.header__sound-img');

document.querySelector('.header__music-btn').addEventListener('click', function () {
    if (isMusicPlaying) {
		audio.play();
        musicIcon.src = "img/music-on.png";
        musicIcon.alt = "Музыка включена";
    } else {
        audio.pause();
        musicIcon.src = "img/music-off.png";
        musicIcon.alt = "Музыка выключена";
    }
    isMusicPlaying = !isMusicPlaying;
});

document.querySelector('.header__sound-btn').addEventListener('click', function () {
    if (isSoundEnabled) {
		soundIcon.src = "img/sound-off.png";
        soundIcon.alt = "Звук выключен";
    } else {
		soundIcon.src = "img/sound-on.png";
        soundIcon.alt = "Звук включен";
    }
    isSoundEnabled = !isSoundEnabled;
});

document.addEventListener('click', function (event) {
    const target = event.target;
    if ((target.tagName === 'BUTTON' || target.tagName === 'A' || target.tagName === 'IMG') 
	&& isSoundEnabled) {
		buttonSound.play();
    }
});