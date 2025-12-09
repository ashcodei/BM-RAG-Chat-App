import React, { useState, useMemo, useRef, useEffect } from "react";

const BACKEND_URL = "http://localhost:8000";

/** ---------- helpers ---------- **/

function makeId(prefix = "id") {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** ---------- message bubble ---------- **/

function MessageBubble({ msg }) {
  const time = new Date(msg.timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  const [attachmentsOpen, setAttachmentsOpen] = useState(false);

  if (msg.role === "user") {
    const attachments = msg.attachments || [];
    const hasAttachments = attachments.length > 0;
    const filesLabel =
      attachments.length === 1
        ? "1 file attached"
        : `${attachments.length} files attached`;

    return (
      <div className="flex justify-end animate-msg">
        <div className="flex items-start gap-3 max-w-[85%]">
          {/* text + bubble column */}
          <div className="flex flex-col items-end gap-1 max-w-full">
            {/* header (time + You) */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-notion-dim">{time}</span>
              <span className="font-medium text-sm">You</span>
            </div>

            {/* bubble */}
            <div className="w-full sm:max-w-[420px]">
              <div className="bg-[#f4f4f3] border border-notion-border rounded-lg px-3 py-2 text-[15px] leading-relaxed text-notion-text shadow-sm">
                {/* attachments header + dropdown */}
                {hasAttachments && (
                  <div className="border-l-2 border-notion-border pl-2 mb-1">
                    <button
                      type="button"
                      onClick={() => setAttachmentsOpen((o) => !o)}
                      className="w-full flex items-center justify-between gap-2 text-left text-[12px] text-notion-dim"
                    >
                      <div className="flex items-center gap-2">
                        {/* small paperclip */}
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="12"
                          height="12"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                        </svg>
                        <span>{filesLabel}</span>
                      </div>

                      {/* small chevron */}
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className={`transition-transform duration-200 ${
                          attachmentsOpen ? "rotate-180" : ""
                        }`}
                      >
                        <polyline points="6 9 12 15 18 9"></polyline>
                      </svg>
                    </button>

                    {/* animated attachments list, shares the same vertical line */}
                    <div
                      className={`
                        overflow-hidden transition-all duration-200 ease-out
                        ${attachmentsOpen ? "max-h-40 mt-2 opacity-100" : "max-h-0 opacity-0"}
                      `}
                    >
                      <div className="flex flex-col gap-1">
                        {attachments.map((att, idx) => (
                          <div
                            key={att.id || idx}
                            className="flex items-center gap-2 bg-white border border-notion-border rounded px-2 py-1 text-[12px] text-notion-text"
                          >
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              width="11"
                              height="11"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="#787774"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
                              <polyline points="14 2 14 8 20 8"></polyline>
                            </svg>
                            <span className="truncate max-w-[220px]">
                              {att.file?.name || "Attachment"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* body text */}
                {msg.text && (
                  <div className={hasAttachments ? "mt-2" : ""}>{msg.text}</div>
                )}
              </div>
            </div>
          </div>

          {/* avatar on the right, not above bubble */}
          <div className="w-8 h-8 rounded-sm bg-notion-text flex items-center justify-center shrink-0 text-white text-sm font-bold">
            U
          </div>
        </div>
      </div>
    );
  }

  // bot
  return (
    <div className="flex gap-4 animate-msg">
      <div className="w-8 h-8 rounded-sm bg-[#e1e0dd] flex items-center justify-center shrink-0 text-sm font-bold text-notion-text">
        A
      </div>
      <div className="flex flex-col gap-1 max-w-[85%]">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">Assistant</span>
          <span className="text-xs text-notion-dim">{time}</span>
        </div>
        <div className="msg-bot text-[15px] leading-relaxed w-full">
          {/* retrieved context */}
          {msg.retrieved && msg.retrieved.length > 0 && (
            <details className="group border-l-2 border-notion-border pl-3 mb-3 open:mb-4 transition-all">
              <summary className="list-none flex items-center gap-2 cursor-pointer text-xs font-medium text-notion-dim hover:text-notion-text transition-colors select-none">
                <svg
                  className="w-4 h-4 transition-transform group-open:rotate-90 text-notion-dim"
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="9 18 15 12 9 6"></polyline>
                </svg>
                Retrieved context
              </summary>
              <div className="mt-3 flex flex-col gap-3 max-h-[300px] overflow-y-auto pr-1">
                {msg.retrieved.map((c, idx) => (
                  <div
                    key={idx}
                    className="bg-notion-sidebar p-3 rounded text-xs text-notion-text border border-notion-border"
                  >
                    <div className="font-semibold mb-1 text-notion-dim flex justify-between">
                      <span>{c.source_pdf}</span>
                      <span className="bg-notion-border px-1 rounded text-[10px]">
                        {typeof c.score === "number"
                          ? `${(c.score * 100).toFixed(1)}% match`
                          : ""}
                      </span>
                    </div>
                    <div className="opacity-90 leading-relaxed font-mono text-[11px] text-[#5a5854]">
                      {c.text.length > 400
                        ? `${c.text.slice(0, 400)}…`
                        : c.text}
                    </div>
                  </div>
                ))}
              </div>
            </details>
          )}
          {/* answer text */}
          <p className="whitespace-pre-wrap">
            {msg.answerText ?? msg.text ?? ""}
          </p>
        </div>
      </div>
    </div>
  );
}

/** ---------- main app ---------- **/

function App() {
  // initial chat
  const initialChatIdRef = useRef(makeId("chat"));
  const initialChatId = initialChatIdRef.current;
  const [fileChunks, setFileChunks] = useState({}); 
  const [chats, setChats] = useState(() => {
    const id = initialChatId;
    return {
      [id]: {
        id,
        title: "Untitled Chat",
        messages: [
          {
            role: "bot",
            text:
              "Hello. I'm ready to help you organize your thoughts or draft content.\n\n" +
              "Upload one or more PDFs above, then ask a question — I will only use the files you attach to that specific message as context.",
            timestamp: Date.now(),
          },
        ],
        library: [], // {id,file,chunks}
        inputQueue: [], // attachments for next message
        createdAt: Date.now(),
      },
    };
  });
  const headerCollapsedRef = useRef(false); 
  const [activeChatId, setActiveChatId] = useState(initialChatId);
  const [view, setView] = useState("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState(null);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [importSelected, setImportSelected] = useState({});
  const [headerCollapsed, setHeaderCollapsed] = useState(false);
  const [inputText, setInputText] = useState("");
  const [isSending, setIsSending] = useState(false);

  const chatContainerRef = useRef(null);
  const textareaRef = useRef(null);
  const autoScrollRef = useRef(false);
  const isContextsView = view === "contexts";
  const activeChat = chats[activeChatId] || null;
  useEffect(() => {
    headerCollapsedRef.current = headerCollapsed;
  }, [headerCollapsed]);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const titleInputRef = useRef(null);
  useEffect(() => {
    setIsEditingTitle(false);
    setTitleDraft("");
  }, [activeChatId]);
  useEffect(() => {
    if (isEditingTitle && titleInputRef.current) {
      titleInputRef.current.focus();
      titleInputRef.current.select();
    }
  }, [isEditingTitle]);

  function buildHistoryForChat(chat) {
    const MAX_MESSAGES = 8; // tweak as needed

    const recent = chat.messages.slice(-MAX_MESSAGES);

    return recent
      .map((m) => {
        const content = (m.answerText ?? m.text ?? "").trim();
        if (!content) return null;
        return {
          role: m.role === "bot" ? "assistant" : "user",
          content,
        };
      })
      .filter(Boolean);
  }

  async function pollIndexStatusForFiles(chatId, fileIdsToNames, maxTries = 20, delayMs = 1500) {
    // fileIdsToNames: { [localFileId]: pdfFileName }
    for (let attempt = 0; attempt < maxTries; attempt++) {
      try {
        const resp = await fetch(`${BACKEND_URL}/index_status`);
        if (!resp.ok) {
          throw new Error(`index_status HTTP ${resp.status}`);
        }
        const data = await resp.json();
        const perPdf = data.per_pdf_counts || {};

        let updatedSomething = false;

        // For each local fileId, see if server now knows the chunk count for that PDF name
        updateChat(chatId, (chat) => {
          let changed = false;
          const newLib = chat.library.map((f) => {
            const pdfName = fileIdsToNames[f.id];
            if (!pdfName) return f;
            const serverCount = perPdf[pdfName];
            if (typeof serverCount === "number" && f.chunks == null) {
              changed = true;
              updatedSomething = true;
              return { ...f, chunks: serverCount };
            }
            return f;
          });
          if (!changed) return chat;
          return { ...chat, library: newLib };
        });

        if (updatedSomething) {
          // Once we updated at least one, we can optionally keep polling to catch others,
          // or just break here. Your call:
          // break;
        }
      } catch (err) {
        console.error("pollIndexStatusForFiles error:", err);
      }

      await new Promise((res) => setTimeout(res, delayMs));
    }
  }

  /** small helper to update a single chat immutably */
  const updateChat = (chatId, updater) => {
    setChats((prev) => {
      const existing = prev[chatId];
      if (!existing) return prev;
      const updated = updater(existing);
      if (updated === existing) return prev;
      return { ...prev, [chatId]: updated };
    });
  };

  const loadChunksForFile = async (pdfName) => {
    setFileChunks((prev) => {
      const current = prev[pdfName];
      if (current && (current.status === "loading" || current.status === "done")) {
        // already loading or loaded; don't refetch
        return prev;
      }
      return {
        ...prev,
        [pdfName]: { status: "loading", chunks: [], error: null },
      };
    });

    try {
      const resp = await fetch(
        `${BACKEND_URL}/file_chunks?pdf=${encodeURIComponent(pdfName)}`
      );
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setFileChunks((prev) => ({
        ...prev,
        [pdfName]: {
          status: "done",
          chunks: data.chunks || [],
          error: null,
        },
      }));
    } catch (err) {
      console.error("file_chunks error:", err);
      setFileChunks((prev) => ({
        ...prev,
        [pdfName]: {
          status: "error",
          chunks: [],
          error: String(err),
        },
      }));
    }
  };

  /** ---------- sidebar / chats ---------- **/

  const sortedChats = useMemo(
    () =>
      Object.values(chats).sort((a, b) => b.createdAt - a.createdAt),
    [chats]
  );

  const createNewChat = () => {
    const id = makeId("chat");
    const newChat = {
      id,
      title: "Untitled Chat",
      messages: [
        {
          role: "bot",
          text:
            "New chat started. Upload PDFs or import from another chat's library, then ask your question.",
          timestamp: Date.now(),
        },
      ],
      library: [],
      inputQueue: [],
      createdAt: Date.now(),
    };
    setChats((prev) => ({ ...prev, [id]: newChat }));
    setActiveChatId(id);
    setView("chat");
  };

  const switchChat = (id) => {
    setActiveChatId(id);
    setView("chat");
  };

  const requestDeleteChat = (id) => {
    setChatToDelete(id);
    setDeleteModalOpen(true);
  };

  const confirmDeleteChat = () => {
    if (!chatToDelete) return;
    setChats((prev) => {
      const copy = { ...prev };
      delete copy[chatToDelete];
      return copy;
    });

    setDeleteModalOpen(false);

    if (activeChatId === chatToDelete) {
      const remainingIds = Object.keys(chats).filter(
        (cid) => cid !== chatToDelete
      );
      if (remainingIds.length > 0) {
        setActiveChatId(remainingIds[0]);
      } else {
        // create fresh chat
        const id = makeId("chat");
        const newChat = {
          id,
          title: "Untitled Chat",
          messages: [
            {
              role: "bot",
              text:
                "Hello. I'm ready to help you organize your thoughts or draft content.\n\nUpload PDFs above, then ask a question.",
              timestamp: Date.now(),
            },
          ],
          library: [],
          inputQueue: [],
          createdAt: Date.now(),
        };
        setChats({ [id]: newChat });
        setActiveChatId(id);
      }
    }

    setChatToDelete(null);
  };

  /** ---------- contexts view ---------- **/

  const allFilesForContexts = useMemo(() => {
    const out = [];
    const signatureSeen = new Set();
    Object.values(chats).forEach((chat) => {
      chat.library.forEach((fileObj) => {
        const sig = fileObj.file.name + "_" + fileObj.file.size;
        if (!signatureSeen.has(sig)) {
          signatureSeen.add(sig);
          out.push({ fileObj, sourceChat: chat.title });
        }
      });
    });
    return out;
  }, [chats]);

  // NEW: deterministic split into two columns
  const { leftFiles, rightFiles } = useMemo(() => {
    const left = [];
    const right = [];
    allFilesForContexts.forEach((item, idx) => {
      (idx % 2 === 0 ? left : right).push(item);
    });
    return { leftFiles: left, rightFiles: right };
  }, [allFilesForContexts]);

  /** ---------- import-from-library ---------- **/

  const availableFilesForImport = useMemo(() => {
    if (!activeChat) return [];
    const currentSig = new Set(
      activeChat.library.map((f) => f.file.name + f.file.size)
    );
    const out = [];
    Object.entries(chats).forEach(([cid, chat]) => {
      if (cid === activeChatId) return;
      chat.library.forEach((f) => {
        const sig = f.file.name + f.file.size;
        if (!currentSig.has(sig)) {
          currentSig.add(sig);
          out.push({ fromChatId: cid, file: f });
        }
      });
    });
    return out;
  }, [chats, activeChat, activeChatId]);

  const handleToggleImportSelection = (fileId, fileObj, checked) => {
    setImportSelected((prev) => {
      const next = { ...prev };
      if (checked) {
        next[fileId] = fileObj;
      } else {
        delete next[fileId];
      }
      return next;
    });
  };

  const importSelectedFiles = () => {
    if (!activeChat) return;
    const files = Object.values(importSelected);
    if (!files.length) {
      setImportModalOpen(false);
      return;
    }

    updateChat(activeChat.id, (chat) => {
      const newLibEntries = files.map((sourceFile) => ({
        id: makeId("f_imp"),
        file: sourceFile.file,
        chunks: sourceFile.chunks,
      }));
      return {
        ...chat,
        library: [...chat.library, ...newLibEntries],
        inputQueue: [...chat.inputQueue, ...newLibEntries],
      };
    });

    setImportSelected({});
    setImportModalOpen(false);
  };

  /** ---------- scroll + header collapse ---------- **/

  useEffect(() => {
    const el = chatContainerRef.current;
    if (!el) return;

    const onScroll = () => {
      if (autoScrollRef.current) return; // ignore programmatic scrolls

      const scrollTop = el.scrollTop;
      const scrollHeight = el.scrollHeight;
      const clientHeight = el.clientHeight;

      const isScrollableEnough = scrollHeight > clientHeight + 150;
      if (!isScrollableEnough) {
        // Not enough content – keep header open
        if (headerCollapsedRef.current) {
          headerCollapsedRef.current = false;
          setHeaderCollapsed(false);
        }
        return;
      }

      const prev = headerCollapsedRef.current;

      // Collapse once user has scrolled a bit
      const COLLAPSE_AT = 140;

      if (!prev && scrollTop > COLLAPSE_AT) {
        headerCollapsedRef.current = true;
        setHeaderCollapsed(true);
        return;
      }
      
      // Only expand when user goes completely back to top
      if (prev && scrollTop === 0) {
        headerCollapsedRef.current = false;
        setHeaderCollapsed(false);
      }
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);


  // Always scroll to bottom when messages change or chat switches
  useEffect(() => {
    const el = chatContainerRef.current;
    if (!el) return;

    autoScrollRef.current = true;
    el.scrollTop = el.scrollHeight;

    requestAnimationFrame(() => {
      autoScrollRef.current = false;
    });
  }, [activeChatId, chats]);
  /** ---------- file handling + upload_pdf ---------- **/

  const handleFiles = async (files) => {
    if (!activeChat) return;

    // Collect which local IDs correspond to which server PDF names
    const fileIdsToNames = {};

    for (const file of files) {
      if (!file.name.toLowerCase().endsWith(".pdf")) continue;

      // check if already in library
      const already = activeChat.library.find(
        (f) => f.file.name === file.name && f.file.size === file.size
      );
      if (already) {
        // just add to input queue
        updateChat(activeChat.id, (chat) => {
          if (chat.inputQueue.some((f) => f.id === already.id)) return chat;
          return {
            ...chat,
            inputQueue: [...chat.inputQueue, already],
          };
        });
        continue;
      }

      const fileId = makeId("f");
      const fileObj = {
        id: fileId,
        file,
        chunks: null, // will be filled in later by polling
      };

      fileIdsToNames[fileId] = file.name;

      // add to library + inputQueue
      updateChat(activeChat.id, (chat) => ({
        ...chat,
        library: [...chat.library, fileObj],
        inputQueue: [...chat.inputQueue, fileObj],
      }));

      // upload to backend (fast response; indexing runs in background)
      try {
        const formData = new FormData();
        formData.append("file", file);
        const resp = await fetch(`${BACKEND_URL}/upload_pdf`, {
          method: "POST",
          body: formData,
        });
        if (!resp.ok) {
          console.error("upload_pdf failed", await resp.text());
          continue;
        }
        // We don't actually need the body except maybe for errors
        await resp.json();
      } catch (err) {
        console.error("Error uploading file:", err);
      }
    }

    // After all uploads, start polling to fill in chunk counts for these files
    const hasNew = Object.keys(fileIdsToNames).length > 0;
    if (hasNew) {
      pollIndexStatusForFiles(activeChat.id, fileIdsToNames);
    }
  };


  /** ---------- library & input queue ---------- **/

  const removeFromLib = (fileId, e) => {
    e.stopPropagation();
    e.preventDefault();
    if (!activeChat) return;
    updateChat(activeChat.id, (chat) => ({
      ...chat,
      library: chat.library.filter((f) => f.id !== fileId),
      inputQueue: chat.inputQueue.filter((f) => f.id !== fileId),
    }));
  };

  const addToInputQueue = (fileObj) => {
    if (!activeChat) return;
    updateChat(activeChat.id, (chat) => {
      if (chat.inputQueue.some((f) => f.id === fileObj.id)) return chat;
      return {
        ...chat,
        inputQueue: [...chat.inputQueue, fileObj],
      };
    });
  };

  const removeFromQueue = (fileId) => {
    if (!activeChat) return;
    updateChat(activeChat.id, (chat) => ({
      ...chat,
      inputQueue: chat.inputQueue.filter((f) => f.id !== fileId),
    }));
  };

  /** ---------- chat -> /chat backend ---------- **/

  const canSend =
    activeChat && (!!inputText.trim() || activeChat.inputQueue.length > 0);

  const sendMessage = async () => {
    if (!activeChat) return;
    if (!canSend || isSending) return;

    const text = inputText.trim();
    const attachments = activeChat.inputQueue;
    const hasAttachments = attachments.length > 0;
    const history = buildHistoryForChat(activeChat);
    const userMsg = {
      role: "user",
      text: text || (hasAttachments ? `[Sent ${attachments.length} file(s)]` : ""),
      attachments: [...attachments],
      timestamp: Date.now(),
    };

    // add user message + clear inputQueue
    updateChat(activeChat.id, (chat) => ({
      ...chat,
      messages: [...chat.messages, userMsg],
    }));

    setInputText("");

    const sources = attachments
      .map((f) => f.file)
      .filter((f) => f && f.name.toLowerCase().endsWith(".pdf"))
      .map((f) => f.name);

    setIsSending(true);

    try {
      const resp = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text || "",
          sources: sources.length ? sources : null, history,
        }),
      });

      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`;
        try {
          const errJson = await resp.json();
          if (errJson.detail) detail = errJson.detail;
        } catch {
          // ignore
        }
        throw new Error(detail);
      }

      const data = await resp.json();
      const botMsg = {
        role: "bot",
        text: "",
        answerText: data.answer ?? "",
        retrieved: data.retrieved ?? [],
        timestamp: Date.now(),
      };

      updateChat(activeChat.id, (chat) => ({
        ...chat,
        messages: [...chat.messages, botMsg],
      }));
    } catch (err) {
      console.error("chat error", err);
      const botMsg = {
        role: "bot",
        text: `Error talking to backend: ${escapeHtml(err.message)}`,
        timestamp: Date.now(),
      };
      updateChat(activeChat.id, (chat) => ({
        ...chat,
        messages: [...chat.messages, botMsg],
      }));
    } finally {
      setIsSending(false);
    }
  };

  /** ---------- UI handlers ---------- **/

  const toggleSidebar = () => {
    setSidebarOpen((s) => !s);
  };

  const handleInputChange = (e) => {
    setInputText(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  const onDropTop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length) handleFiles(files);
  };

  const onDropBottom = (e) => {
    e.preventDefault();
    const fid = e.dataTransfer.getData("fid");
    if (fid && activeChat) {
      const f = activeChat.library.find((x) => x.id === fid);
      if (f) addToInputQueue(f);
      return;
    }
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length) handleFiles(files);
  };

  const onDragStartLibraryItem = (fileId, e) => {
    e.dataTransfer.setData("fid", fileId);
    e.dataTransfer.effectAllowed = "copy";
  };

  const beginEditTitle = () => {
    if (!activeChat) return;
    setTitleDraft(activeChat.title || "Untitled Chat");
    setIsEditingTitle(true);
  };

  const commitTitleEdit = () => {
    if (!activeChat) {
      setIsEditingTitle(false);
      return;
    }
    const newTitle = (titleDraft || "").trim() || "Untitled Chat";

    updateChat(activeChat.id, (chat) => ({
      ...chat,
      title: newTitle,
    }));

    setIsEditingTitle(false);
  };

  const handleTitleKeyDown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitTitleEdit();
    } else if (e.key === "Escape") {
      // cancel & revert
      setTitleDraft(activeChat?.title || "Untitled Chat");
      setIsEditingTitle(false);
    }
  };

  /** ---------- render ---------- **/

  return (
    <div className="h-screen flex overflow-hidden text-notion-text bg-notion-bg text-sm md:text-base selection:bg-[#cce9ff]">
      {/* Sidebar */}
      <aside
        id="sidebar"
        className={`bg-notion-sidebar border-r border-notion-border flex flex-col shrink-0 relative group z-30 ${
          sidebarOpen ? "w-64" : "sidebar-closed"
        }`}
      >
        {/* Sidebar Header */}
        <div className="h-12 flex items-center px-3 border-b border-notion-border shrink-0">
          <div className="flex items-center gap-2 font-medium truncate flex-1 cursor-pointer hover:bg-notion-hover p-1 rounded transition-colors">
            <div className="w-5 h-5 bg-[#e1e0dd] rounded text-[10px] flex items-center justify-center font-bold">
              W
            </div>
            <span className="truncate">My Workspace</span>
          </div>
          <button
            onClick={toggleSidebar}
            className="p-1 rounded hover:bg-notion-hover text-notion-dim transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        {/* Navigation Actions */}
        <div className="p-2 flex flex-col gap-1 shrink-0">
          <button
            onClick={createNewChat}
            className="flex items-center gap-2 text-notion-dim hover:bg-notion-hover hover:text-notion-text px-2 py-2.5 rounded transition-colors text-sm font-medium"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            New Chat
          </button>
          <button
            id="btn-all-contexts"
            onClick={() => setView("contexts")}
            className={
              "flex items-center gap-2 px-2 py-2.5 rounded transition-colors text-sm font-medium " +
              (isContextsView
                ? "bg-notion-select text-notion-text"
                : "text-notion-dim hover:bg-notion-hover hover:text-notion-text")
            }
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {isContextsView ? (
                  // "open" folder feel
                  <path d="m6 14 1.45-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 1.66.9H18a2 2 0 0 1 2 2v2"></path>
                ) : (
                  // closed folder (matches your HTML snippet)
                  <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 2H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2z" />
                )}
            </svg>
            All Contexts
          </button>
        </div>

        {/* History List */}
        <div className="px-3 pt-4 pb-2 text-xs font-semibold text-notion-dim uppercase tracking-wider">
          History
        </div>
        <div
          id="chat-history-list"
          className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5"
        >
          {sortedChats.map((chat) => {
            const isActive = chat.id === activeChatId && view === "chat";
            return (
              <div
                key={chat.id}
                className={`group/item px-2 py-2.5 rounded cursor-pointer text-sm flex items-center justify-between transition-colors ${
                  isActive
                    ? "bg-notion-select text-notion-text font-medium"
                    : "text-notion-dim hover:bg-notion-hover hover:text-notion-text"
                }`}
                onClick={() => switchChat(chat.id)}
              >
                <div className="flex items-center gap-2 truncate">
                  <svg
                    className="shrink-0"
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                  </svg>
                  <span className="truncate">{chat.title}</span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    requestDeleteChat(chat.id);
                  }}
                  className="opacity-0 group-hover/item:opacity-100 hover:text-red-500 transition-opacity p-1"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  </svg>
                </button>
              </div>
            );
          })}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full relative overflow-hidden bg-white">
        {/* VIEW: CHAT */}
        {view === "chat" && activeChat && (
          <div id="view-chat" className="flex flex-col h-full w-full">
            {/* Header */}
            <header className="h-12 border-b border-notion-border flex items-center px-4 justify-between bg-white z-10 shrink-0">
              <div
                id="chat-header-content"
                className="flex items-center gap-2 ml-8 md:ml-0 transition-all duration-300 group/header"
              >
                <div id="sidebar-toggle-overlay" className="z-50">
                  {!sidebarOpen && (<button
                    id="open-sidebar-btn"
                    onClick={toggleSidebar}
                    className="p-1.5 rounded-md hover:bg-notion-hover text-notion-dim transition-colors bg-white shadow-sm border border-notion-border"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <line x1="3" y1="12" x2="21" y2="12"></line>
                      <line x1="3" y1="6" x2="21" y2="6"></line>
                      <line x1="3" y1="18" x2="21" y2="18"></line>
                    </svg>
                  </button>)}
                </div>
                {/* Title edit container */}
                
                <div className="relative flex items-center gap-1">
                  {isEditingTitle ? (
                    <input
                      ref={titleInputRef}
                      type="text"
                      className="font-medium text-notion-text bg-transparent outline-none border-b border-notion-text/30 w-full min-w-[150px] max-w-[300px] text-sm md:text-base pb-0.5"
                      spellCheck={false}
                      autoComplete="off"
                      value={titleDraft}
                      onChange={(e) => setTitleDraft(e.target.value)}
                      onBlur={commitTitleEdit}
                      onKeyDown={handleTitleKeyDown}
                    />
                  ) : (
                    <>
                      <span
                        id="chat-title"
                        className="font-medium truncate max-w-[200px] md:max-w-md cursor-text border-b border-transparent hover:border-notion-border/50 transition-colors"
                        onClick={beginEditTitle}
                      >
                        {activeChat.title}
                      </span>
                      <button
                        type="button"
                        onClick={beginEditTitle}
                        className="opacity-0 group-hover/header:opacity-100 text-notion-dim hover:text-notion-text transition-all p-1 rounded hover:bg-notion-hover ml-1"
                        title="Rename Chat"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="15"
                          height="15"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                        </svg>
                      </button>
                    </>
                  )}
                </div>

                <span className="text-notion-dim text-xs px-1.5 py-0.5 rounded border border-notion-border select-none">
                  Chat
                </span>
              </div>
            </header>

            {/* Upload Zone */}
            <div
              id="upload-zone"
              className={`bg-[#fbfbfa] border-b border-notion-border px-4 py-3 shrink-0 transition-all duration-300 z-20 ${
                headerCollapsed ? "collapsed-zone" : ""
              }`}
            >
              <div className="mx-auto flex flex-col">
                {/* Top row */}
                <div
                  id="top-actions-row"
                  className="flex items-center justify-between mb-2 overflow-hidden"
                >
                  <span className="text-xs font-semibold text-notion-dim uppercase tracking-wider">
                    Library
                  </span>
                  <button
                    onClick={() => setImportModalOpen(true)}
                    className="text-xs flex items-center gap-1 text-notion-dim hover:text-notion-text transition-colors px-2 py-1 rounded hover:bg-notion-hover"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Import from Library
                  </button>
                </div>

                {/* Large Drop Zone */}
                <div
                  id="large-drop-wrapper"
                  className="overflow-hidden transition-all duration-300 max-h-[120px] opacity-100"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={onDropTop}
                >
                  <label
                    htmlFor="file-upload"
                    className="group flex items-center gap-3 p-3 border border-dashed border-[#d3d1cb] rounded-md cursor-pointer hover:bg-notion-hover hover:border-notion-dim transition-all relative overflow-hidden mb-1"
                  >
                    <div className="bg-white p-2 rounded shadow-sm border border-notion-border group-hover:shadow transition-shadow">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="20"
                        height="20"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="#9b9a97"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="group-hover:stroke-notion-text transition-colors"
                      >
                        <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                      </svg>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-notion-text">
                        Click to upload files
                      </span>
                      <span className="text-xs text-notion-dim">
                        Files added here are specific to this chat session.
                      </span>
                    </div>
                  </label>
                </div>

                {/* Compact toolbar + library list */}
                <div className="flex items-center w-full">
                  {/* mini upload */}
                  <label
                    id="mini-upload-btn"
                    htmlFor="file-upload"
                    className="flex items-center justify-center h-8 bg-white border border-notion-border rounded hover:bg-notion-hover cursor-pointer text-notion-dim hover:text-notion-text shrink-0 w-0 opacity-0 overflow-hidden transition-all duration-300 mr-0"
                    title="Upload File"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                    </svg>
                  </label>

                  {/* mini import */}
                  <button
                    id="mini-import-btn"
                    onClick={() => setImportModalOpen(true)}
                    className="flex items-center justify-center h-8 bg-notion-text rounded hover:opacity-90 cursor-pointer text-white shrink-0 w-0 opacity-0 overflow-hidden transition-all duration-300 mr-0"
                    title="Import from Library"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                  </button>

                  {/* library */}
                  <div
                    id="library-container"
                    className={`w-full overflow-hidden relative ${
                      activeChat.library.length === 0 ? "hidden" : ""
                    }`}
                  >
                    <div
                      id="lib-left-ctrl"
                      className="absolute left-0 top-0 bottom-0 flex items-center pl-1 pr-6 bg-gradient-to-r from-[#fbfbfa] via-[#fbfbfa] to-transparent z-10 opacity-0 pointer-events-none transition-opacity duration-200"
                    >
                      <button
                        onClick={() =>
                          (document.getElementById("library-list").scrollLeft -=
                            200)
                        }
                        className="w-6 h-6 bg-notion-text rounded-full shadow-sm flex items-center justify-center text-white hover:opacity-80 transition-opacity pointer-events-auto"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="14"
                          height="14"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polyline points="15 18 9 12 15 6"></polyline>
                        </svg>
                      </button>
                    </div>
                    <div
                      id="library-list"
                      className="flex flex-nowrap gap-2 overflow-x-auto no-scrollbar items-center h-full scroll-smooth pr-10"
                    >
                      {activeChat.library.map((item) => {
                        const meta =
                          item.chunks == null ? (
                            <svg
                              className="animate-spin h-3 w-3 text-notion-dim ml-1.5"
                              xmlns="http://www.w3.org/2000/svg"
                              fill="none"
                              viewBox="0 0 24 24"
                            >
                              <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                              ></circle>
                              <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                              ></path>
                            </svg>
                          ) : (
                            <span className="font-bold text-[10px] ml-1.5">
                              {item.chunks}
                            </span>
                          );
                        return (
                          <div
                            key={item.id}
                            className="shrink-0 draggable-source flex items-center gap-2 bg-white border border-notion-border pl-2 pr-1.5 py-1.5 rounded-md text-xs text-notion-text shadow-sm group select-none cursor-grab active:cursor-grabbing"
                            draggable
                            onDragStart={(e) =>
                              onDragStartLibraryItem(item.id, e)
                            }
                          >
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              width="14"
                              height="14"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="9B9A97"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
                              <polyline points="14 2 14 8 20 8"></polyline>
                            </svg>
                            <span className="max-w-[150px] truncate">
                              {item.file.name}
                            </span>
                            {meta}
                            <button
                              className="text-[#dfdfdd] hover:text-red-500 transition-colors ml-1"
                              onClick={(e) => removeFromLib(item.id, e)}
                            >
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="14"
                                height="14"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <line
                                  x1="18"
                                  y1="6"
                                  x2="6"
                                  y2="18"
                                ></line>
                                <line
                                  x1="6"
                                  y1="6"
                                  x2="18"
                                  y2="18"
                                ></line>
                              </svg>
                            </button>
                          </div>
                        );
                      })}
                    </div>
                    <div
                      id="lib-right-ctrl"
                      className="absolute right-0 top-0 bottom-0 flex items-center pr-1 pl-6 bg-gradient-to-l from-[#fbfbfa] via-[#fbfbfa] to-transparent z-10 opacity-0 pointer-events-none transition-opacity duration-200"
                    >
                      <button
                        onClick={() =>
                          (document.getElementById("library-list").scrollLeft +=
                            200)
                        }
                        className="w-6 h-6 bg-notion-text rounded-full shadow-sm flex items-center justify-center text-white hover:opacity-80 transition-opacity pointer-events-auto"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="14"
                          height="14"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polyline points="9 18 15 12 9 6"></polyline>
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>

                <input
                  id="file-upload"
                  type="file"
                  className="hidden"
                  multiple
                  onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    if (files.length) handleFiles(files);
                    e.target.value = "";
                  }}
                />
              </div>
            </div>

            {/* Messages */}
            <div
              id="chat-container"
              ref={chatContainerRef}
              className="flex-1 overflow-y-auto px-4 py-6 scroll-smooth"
            >
              <div
                className="mx-auto flex flex-col gap-6"
                id="message-list"
              >
                {activeChat.messages.map((m, idx) => (
                  <MessageBubble key={idx} msg={m} />
                ))}

                {isSending && (
                  <div className="flex gap-4 animate-msg" id="temp-loader">
                    <div className="w-8 h-8 rounded-sm bg-[#e1e0dd] flex items-center justify-center shrink-0 text-sm font-bold text-notion-text">
                      A
                    </div>
                    <div className="flex flex-col gap-1 justify-center">
                      <div className="shimmer">Searching knowledge base...</div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Input footer */}
            <div className="p-4 bg-white/90 backdrop-blur-sm border-t border-notion-border shrink-0">
              <div className="mx-auto relative flex flex-col gap-2">
                {/* attachment queue */}
                <div
                  id="input-queue-wrapper"
                  className={`relative w-full overflow-hidden ${
                    activeChat.inputQueue.length === 0 ? "hidden" : ""
                  }`}
                >
                  <div
                    id="input-left-ctrl"
                    className="absolute left-0 top-0 bottom-0 flex items-center pl-1 pr-6 bg-gradient-to-r from-white via-white to-transparent z-10 opacity-0 pointer-events-none transition-opacity duration-200"
                  >
                    <button
                      onClick={() =>
                        (document.getElementById(
                          "input-file-queue"
                        ).scrollLeft -= 150)
                      }
                      className="w-5 h-5 bg-notion-text rounded-full shadow-sm flex items-center justify-center text-white hover:opacity-80 transition-opacity pointer-events-auto"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="15 18 9 12 15 6"></polyline>
                      </svg>
                    </button>
                  </div>
                  <div
                    id="input-file-queue"
                    className="flex flex-nowrap gap-2 overflow-x-auto no-scrollbar items-center scroll-smooth py-1 pr-10"
                  >
                    {activeChat.inputQueue.map((item) => (
                      <div
                        key={item.id}
                        className="shrink-0 flex items-center gap-2 bg-[#f0f0ee] border border-notion-border pl-2 pr-1.5 py-1 rounded-full text-xs text-notion-text"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="12"
                          height="12"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="#787774"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
                          <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                        <span className="max-w-[120px] truncate">
                          {item.file.name}
                        </span>
                        <button
                          onClick={() => removeFromQueue(item.id)}
                          className="text-notion-dim hover:text-notion-text transition-colors ml-1"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="12"
                            height="12"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                  <div
                    id="input-right-ctrl"
                    className="absolute right-0 top-0 bottom-0 flex items-center pr-1 pl-6 bg-gradient-to-l from-white via-white to-transparent z-10 opacity-0 pointer-events-none transition-opacity duration-200"
                  >
                    <button
                      onClick={() =>
                        (document.getElementById(
                          "input-file-queue"
                        ).scrollLeft += 150)
                      }
                      className="w-5 h-5 bg-notion-text rounded-full shadow-sm flex items-center justify-center text-white hover:opacity-80 transition-opacity pointer-events-auto"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="9 18 15 12 9 6"></polyline>
                      </svg>
                    </button>
                  </div>
                </div>

                <div
                  id="input-drop-zone"
                  className="relative flex items-end gap-2 border border-notion-border rounded-lg bg-white shadow-sm focus-within:ring-1 focus-within:ring-[#d3d1cb] focus-within:border-notion-dim transition-all p-2"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={onDropBottom}
                >
                  <textarea
                    id="chat-input"
                    ref={textareaRef}
                    rows={1}
                    placeholder="Type a message... (Drag files here to attach)"
                    className="w-full resize-none border-none outline-none text-[15px] max-h-32 bg-transparent py-2 px-1 placeholder:text-notion-dim"
                    style={{ minHeight: "40px" }}
                    value={inputText}
                    onChange={handleInputChange}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                  />
                  <button
                    id="send-btn"
                    className={`p-2 rounded hover:bg-notion-hover transition-colors mb-0.5 ${
                      canSend
                        ? "text-notion-text"
                        : "text-notion-dim disabled:opacity-50 disabled:cursor-not-allowed"
                    }`}
                    disabled={!canSend || isSending}
                    onClick={sendMessage}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <line x1="22" y1="2" x2="11" y2="13"></line>
                      <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* VIEW: CONTEXTS */}
        {view === "contexts" && (
          <div
            id="view-contexts"
            className="flex flex-col h-full w-full bg-white overflow-y-auto"
          >
            <header className="h-12 border-b border-notion-border flex items-center px-4 justify-between bg-white sticky top-0 z-10">
              <div className="flex items-center gap-2 ml-8 md:ml-0">
                <div id="sidebar-toggle-overlay" className="z-50">
                  {!sidebarOpen && (<button
                    id="open-sidebar-btn"
                    onClick={toggleSidebar}
                    className="p-1.5 rounded-md hover:bg-notion-hover text-notion-dim transition-colors bg-white shadow-sm border border-notion-border"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <line x1="3" y1="12" x2="21" y2="12"></line>
                      <line x1="3" y1="6" x2="21" y2="6"></line>
                      <line x1="3" y1="18" x2="21" y2="18"></line>
                    </svg>
                  </button>)}
                </div>
                <span className="font-medium">All Contexts</span>
                <span
                  id="total-files-badge"
                  className="text-notion-dim text-xs px-1.5 py-0.5 rounded border border-notion-border"
                >
                  {allFilesForContexts.length} Files
                </span>
              </div>
            </header>
            <div className="p-8 max-w-5xl mx-auto w-full">
              <div className="mb-6 text-sm text-notion-dim">
                Files uploaded across all your chats are accessible here. This
                is a high-level view; retrieval in chat uses the actual stored
                chunks.
              </div>

              <div
                className="mb-6"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const files = Array.from(e.dataTransfer.files || []);
                  if (files.length) handleFiles(files); // reuses your existing uploader
                }}
              >
                <label
                  htmlFor="contexts-file-upload"
                  className="group flex items-center gap-3 p-3 border border-dashed border-[#d3d1cb] rounded-md cursor-pointer hover:bg-notion-hover hover:border-notion-dim transition-all"
                >
                  <div className="bg-white p-2 rounded shadow-sm border border-notion-border group-hover:shadow">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="#9b9a97"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="group-hover:stroke-notion-text transition-colors"
                    >
                      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                    </svg>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-sm font-medium text-notion-text">
                      Click to upload files
                    </span>
                    <span className="text-xs text-notion-dim">
                      Files uploaded here are available in all chats and will be auto-added to your current chat&apos;s library.
                    </span>
                  </div>
                </label>
                <input
                  id="contexts-file-upload"
                  type="file"
                  className="hidden"
                  multiple
                  onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    if (files.length) handleFiles(files);
                    e.target.value = "";
                  }}
                />
              </div>
              <div
                id="contexts-grid"
                className="columns-1 md:columns-2 gap-4"
              >
                {allFilesForContexts.length === 0 && (
                  <div className="col-span-full text-center text-notion-dim py-12">
                    No files uploaded yet.
                  </div>
                )}
                {allFilesForContexts.map(({ fileObj, sourceChat }, idx) => {
                  const isLoading =
                    fileObj.chunks === null || fileObj.chunks === undefined;
                  const chunkCount = isLoading
                    ? "Processing..."
                    : `${fileObj.chunks} Chunks`;
                  const sizeStr =
                    (fileObj.file.size / 1024).toFixed(1) + " KB";
                  return (
                    <div
                      key={fileObj.id + "_" + idx}
                      className="border border-notion-border rounded-lg bg-[#fbfbfa] overflow-hidden hover:shadow-sm transition-shadow mb-4"
                      style={{breakInside: "avoid"}}
                    >
                      <details
                        className="group"
                        onToggle={(e) => {
                          if (e.target.open) {
                            loadChunksForFile(fileObj.file.name);
                          }
                        }}
                      >
                        <summary className="flex items-center gap-3 p-4 cursor-pointer select-none bg-white border-b border-transparent group-open:border-notion-border transition-colors">
                          {/* existing summary unchanged */}
                          <div className="w-10 h-10 rounded bg-[#f1f1ef] flex items-center justify-center text-notion-dim shrink-0">
                            {isLoading ? (
                              <svg
                                className="animate-spin h-5 w-5"
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                              >
                                <circle
                                  className="opacity-25"
                                  cx="12"
                                  cy="12"
                                  r="10"
                                  stroke="currentColor"
                                  strokeWidth="4"
                                ></circle>
                                <path
                                  className="opacity-75"
                                  fill="currentColor"
                                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                ></path>
                              </svg>
                            ) : (
                              <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="20"
                                height="20"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
                                <polyline points="14 2 14 8 20 8"></polyline>
                              </svg>
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="font-medium truncate text-notion-text">
                              {fileObj.file.name}
                            </div>
                            <div className="text-xs text-notion-dim flex items-center gap-2">
                              <span>{sizeStr}</span>
                              <span>•</span>
                              <span className={isLoading ? "italic" : "font-semibold"}>
                                {chunkCount}
                              </span>
                            </div>
                          </div>
                          <svg
                            className="w-5 h-5 text-notion-dim transition-transform group-open:rotate-180"
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <polyline points="6 9 12 15 18 9"></polyline>
                          </svg>
                        </summary>

                        {/* Chunks panel */}
                        <div className="p-4 bg-[#fbfbfa] text-sm space-y-3 max-h-[260px] overflow-y-auto no-scrollbar">
                          {(() => {
                            const fc = fileChunks[fileObj.file.name];
                            if (!fc || fc.status === "loading") {
                              return (
                                <div className="text-notion-dim text-xs italic shimmer">
                                  Loading chunks for this file...
                                </div>
                              );
                            }
                            if (fc.status === "error") {
                              return (
                                <div className="text-red-500 text-xs">
                                  Error loading chunks: {fc.error}
                                </div>
                              );
                            }
                            if (!fc.chunks.length) {
                              return (
                                <div className="text-notion-dim text-xs">
                                  No chunks found for this file.
                                </div>
                              );
                            }
                            return (
                              <div className="flex flex-col gap-3">
                                {fc.chunks.map((chunk) => (
                                  <div
                                    key={chunk.id}
                                    className="border border-notion-border rounded-md bg-white p-2"
                                  >
                                    <div className="text-[10px] text-notion-dim mb-1">
                                      Chunk {chunk.chunk_index} (id {chunk.id})
                                    </div>
                                    <div className="text-xs leading-relaxed whitespace-pre-wrap">
                                      {(() => {
                                        // take first ~2 lines worth of text
                                        const maxChars = 200; // tune this to taste
                                        let preview = chunk.text || "";

                                        // Optional: be nicer about line breaks
                                        const lines = preview.split("\n").filter(Boolean);
                                        if (lines.length > 2) {
                                          preview = lines.slice(0, 2).join(" ");
                                        }
                                        preview = preview.trim() + "…";

                                        return preview;
                                      })()}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            );
                          })()}
                          <div className="text-[10px] text-right text-notion-dim mt-2">
                            Source chat: {sourceChat}
                          </div>
                        </div>
                      </details>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Delete modal */}
      {deleteModalOpen && (
        <div id="delete-modal" className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/20 backdrop-blur-sm"
            onClick={() => setDeleteModalOpen(false)}
          ></div>
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl border border-notion-border w-full max-w-sm flex flex-col">
            <div className="p-4">
              <h3 className="font-medium text-notion-text mb-2">
                Delete Chat?
              </h3>
              <p className="text-sm text-notion-dim">
                This action cannot be undone. All messages and file associations
                in this chat will be removed.
              </p>
            </div>
            <div className="p-3 border-t border-notion-border bg-white flex justify-end gap-2 rounded-b-lg">
              <button
                onClick={() => setDeleteModalOpen(false)}
                className="px-3 py-1.5 text-sm text-notion-dim hover:bg-notion-hover rounded transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteChat}
                className="px-3 py-1.5 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Import modal */}
      {importModalOpen && (
        <div id="import-modal" className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/20 backdrop-blur-sm"
            onClick={() => {
              setImportModalOpen(false);
              setImportSelected({});
            }}
          ></div>
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl border border-notion-border w-full max-w-md flex flex-col max-h-[80vh]">
            <div className="p-4 border-b border-notion-border flex items-center justify-between">
              <h3 className="font-medium text-notion-text">
                Import from Library
              </h3>
              <button
                onClick={() => {
                  setImportModalOpen(false);
                  setImportSelected({});
                }}
                className="text-notion-dim hover:text-notion-text"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>
            <div
              className="p-2 overflow-y-auto flex-1 bg-[#fbfbfa]"
              id="import-list"
            >
              {availableFilesForImport.length === 0 ? (
                <div className="text-center text-notion-dim py-8 text-sm">
                  No other files available to import.
                </div>
              ) : (
                availableFilesForImport.map(({ file }) => (
                  <label
                    key={file.id}
                    className="flex items-center gap-3 p-3 bg-white border border-notion-border rounded mb-2 cursor-pointer hover:bg-[#f7f7f5]"
                  >
                    <input
                      type="checkbox"
                      className="w-4 h-4 rounded border-gray-300 text-notion-text focus:ring-notion-dim"
                      checked={!!importSelected[file.id]}
                      onChange={(e) =>
                        handleToggleImportSelection(
                          file.id,
                          file,
                          e.target.checked
                        )
                      }
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-notion-text truncate">
                        {file.file.name}
                      </div>
                      <div className="text-xs text-notion-dim">
                        {(file.file.size / 1024).toFixed(1)} KB •{" "}
                        {file.chunks ?? "?"} chunks
                      </div>
                    </div>
                  </label>
                ))
              )}
            </div>
            <div className="p-3 border-t border-notion-border bg-white flex justify-end gap-2">
              <button
                onClick={() => {
                  setImportModalOpen(false);
                  setImportSelected({});
                }}
                className="px-3 py-1.5 text-sm text-notion-dim hover:bg-notion-hover rounded transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={importSelectedFiles}
                className="px-3 py-1.5 text-sm bg-notion-text text-white rounded hover:opacity-90 transition-opacity"
              >
                Import Selected
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
