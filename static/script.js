const chatBox = document.getElementById("chat-box");
const input = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

// Initial welcome trigger
window.onload = () => {
    sendToBot("start");
};

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
});

function sendMessage() {
    const msg = input.value.trim();
    if (!msg) return;
    addMessage(msg, "user");
    input.value = "";
    sendToBot(msg);
}

function sendToBot(message) {
    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    })
    .then(res => res.json())
    .then(data => {
        addMessage(data.reply, "bot");
    })
    .catch(() => {
        addMessage("⚠️ Server error. Try again.", "bot");
    });
}

function addMessage(text, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${sender}`;
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
