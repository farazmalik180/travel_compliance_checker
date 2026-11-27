// PIM Frontend Script connecting to AWS Lambda Backend
const BACKEND_URL = "https://uplfb4vsok54ao32lqrbagm56a0wtcdh.lambda-url.us-east-1.on.aws";

document.addEventListener("DOMContentLoaded", () => {
  const complianceForm = document.getElementById("complianceForm");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  const resultsContainer = document.getElementById("resultsContainer");
  const evalBadge = document.getElementById("evalBadge");
  const statusBanner = document.getElementById("statusBanner");
  const statusIcon = document.getElementById("statusIcon");
  const statusTitle = document.getElementById("statusTitle");
  const statusSubtitle = document.getElementById("statusSubtitle");
  const scoreText = document.getElementById("scoreText");
  const verifiedList = document.getElementById("verifiedList");
  const missingList = document.getElementById("missingList");
  const advocateNotes = document.getElementById("advocateNotes");
  const ruleReference = document.getElementById("ruleReference");

  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const chatMessages = document.getElementById("chatMessages");
  const promptChips = document.querySelectorAll(".prompt-chip");

  let conversationHistory = [];

  // Handle Compliance Check Submission
  complianceForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
      nationality: document.getElementById("nationality").value,
      destination: document.getElementById("destination").value,
      visa_category: document.getElementById("visaCategory").value,
      purpose: document.getElementById("purpose").value,
      passport_history: document.getElementById("passportHistory").value,
      profession: document.getElementById("profession").value,
      bank_funds: document.getElementById("bankFunds").value,
      has_return_ticket: document.getElementById("hasReturnTicket").checked,
      has_hotel_booking: document.getElementById("hasHotelBooking").checked,
      has_financial_proof: document.getElementById("hasFinancialProof").checked,
      has_protector_stamp: document.getElementById("hasProtectorStamp").checked
    };

    // UI Loading state
    resultsPlaceholder.classList.add("d-none");
    resultsContainer.classList.add("d-none");
    loadingSpinner.classList.remove("d-none");
    evalBadge.className = "badge bg-warning text-dark";
    evalBadge.textContent = "Adjudicating...";

    try {
      const resp = await fetch(`${BACKEND_URL}/api/check-compliance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        throw new Error(`HTTP Error ${resp.status}`);
      }

      const data = await resp.json();
      renderComplianceResults(data);
    } catch (err) {
      alert(`Error connecting to compliance engine: ${err.message}`);
      resultsPlaceholder.classList.remove("d-none");
      evalBadge.className = "badge bg-danger";
      evalBadge.textContent = "Error";
    } finally {
      loadingSpinner.classList.add("d-none");
    }
  });

  function renderComplianceResults(data) {
    resultsContainer.classList.remove("d-none");

    const isCleared = data.status === "CLEARED";

    // Set Status Banner
    if (isCleared) {
      evalBadge.className = "badge bg-success";
      evalBadge.textContent = "CLEARED";
      statusBanner.className = "alert alert-success d-flex align-items-center justify-content-between p-3 mb-4 rounded-3 border-0";
      statusIcon.className = "bi bi-check-circle-fill text-success display-5";
      statusTitle.textContent = "DEPARTURE CLEARED";
      statusTitle.className = "h5 mb-1 fw-bold text-success";
      statusSubtitle.textContent = "Passenger documentation satisfies FIA departure requirements.";
    } else {
      evalBadge.className = "badge bg-danger";
      evalBadge.textContent = "OFF-LOAD RISK";
      statusBanner.className = "alert alert-danger d-flex align-items-center justify-content-between p-3 mb-4 rounded-3 border-0";
      statusIcon.className = "bi bi-x-octagon-fill text-danger display-5";
      statusTitle.textContent = "OFF-LOAD RISK FLAGGED";
      statusTitle.className = "h5 mb-1 fw-bold text-danger";
      statusSubtitle.textContent = "Action required: Statutory or profile deficiencies identified.";
    }

    scoreText.textContent = data.compliance_score || "N/A";

    // Render Verified Items
    verifiedList.innerHTML = "";
    if (data.verified_items && data.verified_items.length > 0) {
      data.verified_items.forEach(item => {
        const li = document.createElement("li");
        li.className = "list-group-item bg-transparent px-0 py-1 d-flex align-items-start gap-2 text-secondary";
        li.innerHTML = `<i class="bi bi-check2 text-success fs-6 mt-1"></i> <span>${item}</span>`;
        verifiedList.appendChild(li);
      });
    } else {
      verifiedList.innerHTML = '<li class="list-group-item bg-transparent text-muted small">No verified requirements logged.</li>';
    }

    // Render Missing Items
    missingList.innerHTML = "";
    if (data.missing_or_incomplete_requirements && data.missing_or_incomplete_requirements.length > 0) {
      data.missing_or_incomplete_requirements.forEach(item => {
        const li = document.createElement("li");
        li.className = "list-group-item bg-transparent px-0 py-1 d-flex align-items-start gap-2 text-danger fw-semibold";
        li.innerHTML = `<i class="bi bi-exclamation-circle text-danger fs-6 mt-1"></i> <span>${item}</span>`;
        missingList.appendChild(li);
      });
    } else {
      missingList.innerHTML = '<li class="list-group-item bg-transparent text-success small"><i class="bi bi-check-all me-1"></i>No missing items! All critical requirements met.</li>';
    }

    // Advocate Notes
    advocateNotes.textContent = data.advocate_notes || "No additional advocate observations.";

    // Rule Reference
    ruleReference.textContent = data.fia_rule_reference || "Standard FIA Integrated Border Management System (IBMS) guidelines.";
  }

  // Handle Chat Submission
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    appendChatMessage("user", query);
    chatInput.value = "";

    conversationHistory.push({ role: "user", content: query });

    // Show temporary thinking message
    const tempId = "temp-msg-" + Date.now();
    appendChatMessage("assistant", "Thinking...", tempId);

    try {
      const resp = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: conversationHistory })
      });

      const data = await resp.json();
      const reply = data.reply || "Sorry, no response received.";

      // Replace temporary message
      const tempElement = document.getElementById(tempId);
      if (tempElement) {
        tempElement.innerHTML = `<strong>PIM Assistant:</strong> ${formatMarkdown(reply)}`;
      }
      conversationHistory.push({ role: "assistant", content: reply });
    } catch (err) {
      const tempElement = document.getElementById(tempId);
      if (tempElement) {
        tempElement.innerHTML = `<strong>PIM Assistant:</strong> Error: ${err.message}`;
      }
    }
  });

  function appendChatMessage(role, text, id = null) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble ${role === "user" ? "user-bubble" : "assistant-bubble"} mb-2`;
    if (id) bubble.id = id;

    if (role === "user") {
      bubble.textContent = text;
    } else {
      bubble.innerHTML = `<strong>PIM Assistant:</strong> ${formatMarkdown(text)}`;
    }

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function formatMarkdown(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
  }

  // Prompt chips
  promptChips.forEach(chip => {
    chip.addEventListener("click", () => {
      chatInput.value = chip.textContent;
      chatForm.dispatchEvent(new Event("submit"));
    });
  });
});
