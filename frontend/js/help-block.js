document.querySelector('.header__help-btn').addEventListener('click', function() {
    const helpBlock = document.querySelector('.help-block');
    
    if (helpBlock.style.display === 'flex') {
        helpBlock.style.opacity = '0';
        setTimeout(() => {
            helpBlock.style.display = 'none';
        }, 300);
    } 

    else {
        helpBlock.style.display = 'flex';
        helpBlock.style.opacity = '0';
        
        setTimeout(() => {
            helpBlock.style.opacity = '1';
            helpBlock.style.transition = 'opacity 0.3s ease-in';
        }, 10);
    }
});