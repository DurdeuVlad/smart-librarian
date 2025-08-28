import React, { useState, useRef, useEffect } from 'react';
import {
    Container,
    Paper,
    TextField,
    Button,
    Typography,
    Box,
    Card,
    CardContent,
    Chip,
    Alert,
    CircularProgress,
    AppBar,
    Toolbar,
    IconButton,
    List,
    ListItem,
    ListItemText,
    Divider,
    Grid,
    LinearProgress
} from '@mui/material';
import BookCover from "./BookCover";
import {
    Send,
    Book,
    AccountBalance,
    AutoStories,
    Search,
    Psychology
} from '@mui/icons-material';

const API_BASE = 'http://localhost:3001/api';

function SmartLibrarian() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [budget, setBudget] = useState({ spent: 0, remaining: 5, limit: 5 });
    const [searchResults, setSearchResults] = useState([]);
    const messagesEndRef = useRef(null);

    // Mock data for development
    const mockBooks = [
        { title: "Maitreyi", authors: "Mircea Eliade", year: 1933, subjects: "romance, philosophy" },
        { title: "Ion", authors: "Liviu Rebreanu", year: 1920, subjects: "realism, drama" },
        { title: "Harry Potter", authors: "J.K. Rowling", year: 1997, subjects: "fantasy, magic" },
        { title: "1984", authors: "George Orwell", year: 1949, subjects: "dystopia, surveillance" }
    ];

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMessage = input.trim();
        setInput('');
        setLoading(true);

        // Adaugă mesajul userului în chat
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

        try {
            const resp = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessage })
            });

            const data = await resp.json();

            if (resp.ok) {
                setMessages(prev => [...prev, { role: 'assistant', content: data.message }]);
                setSearchResults(data.searchResults || []);
                setBudget(data.budget);
            } else {
                setMessages(prev => [...prev, { role: 'assistant', content: data.error || 'Eroare server' }]);
            }
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', content: 'Eroare la conectarea cu serverul.' }]);
        } finally {
            setLoading(false);
        }
    };

    async function generateCover(title, author, summary) {
        const resp = await fetch("http://localhost:3001/api/cover", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, author, summary })
        });
        return resp.json();
    }


    const budgetPercentage = (budget.spent / budget.limit) * 100;

    return (
        <Box sx={{ height: '100vh', width: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <AppBar position="static" sx={{ backgroundColor: '#1976d2' }}>
                <Toolbar>
                    <AutoStories sx={{ mr: 2 }} />
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        Smart Librarian
                    </Typography>
                    <Chip
                        icon={<Psychology />}
                        label={`$${budget.spent.toFixed(2)}/$${budget.limit}`}
                        color={budgetPercentage > 80 ? "error" : "success"}
                        variant="outlined"
                        sx={{ color: 'white', borderColor: 'white' }}
                    />
                </Toolbar>
                <LinearProgress
                    variant="determinate"
                    value={budgetPercentage}
                    sx={{ height: 4 }}
                    color={budgetPercentage > 80 ? "error" : "success"}
                />
            </AppBar>

            <Container maxWidth="lg" sx={{ flex: 1, display: 'flex', py: 2, gap: 2 }}>
                {/* Main Chat */}
                <Box sx={{ flex: 2 }}>
                    <Paper sx={{ height: 'calc(100vh - 200px)', display: 'flex', flexDirection: 'column' }}>
                        {/* Messages */}
                        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
                            {messages.length === 0 && (
                                <Box sx={{ textAlign: 'center', py: 4 }}>
                                    <Book sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
                                    <Typography variant="h5" gutterBottom>
                                        Bun venit la Smart Librarian!
                                    </Typography>
                                    <Typography variant="body1" color="text.secondary">
                                        Întreabă-mă despre cărți, autori sau subiecte care te interesează.
                                    </Typography>
                                    <Box sx={{ mt: 2 }}>
                                        <Chip label="Exemple:" sx={{ mr: 1, mb: 1 }} />
                                        <Chip label="Vreau o carte despre dragoste" variant="outlined" sx={{ mr: 1, mb: 1 }} />
                                        <Chip label="Recomandă-mi fantasy" variant="outlined" sx={{ mr: 1, mb: 1 }} />
                                        <Chip label="Ce este Ion?" variant="outlined" sx={{ mb: 1 }} />
                                    </Box>
                                </Box>
                            )}

                            {messages.map((message, index) => (
                                <Box
                                    key={index}
                                    sx={{
                                        display: 'flex',
                                        justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                                        mb: 2
                                    }}
                                >
                                    <Paper
                                        sx={{
                                            p: 2,
                                            maxWidth: '70%',
                                            backgroundColor: message.role === 'user' ? 'primary.main' : 'grey.100',
                                            color: message.role === 'user' ? 'white' : 'text.primary'
                                        }}
                                    >
                                        <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                                            {message.content}
                                        </Typography>
                                    </Paper>
                                </Box>
                            ))}

                            {loading && (
                                <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                                    <Paper sx={{ p: 2, backgroundColor: 'grey.100' }}>
                                        <CircularProgress size={20} sx={{ mr: 1 }} />
                                        <Typography variant="body2" component="span">
                                            Caut în biblioteca...
                                        </Typography>
                                    </Paper>
                                </Box>
                            )}

                            <div ref={messagesEndRef} />
                        </Box>

                        {/* Input */}
                        <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                            <form onSubmit={handleSubmit}>
                                <TextField
                                    fullWidth
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    placeholder="Întreabă despre o carte..."
                                    disabled={loading}
                                    InputProps={{
                                        endAdornment: (
                                            <IconButton type="submit" disabled={loading || !input.trim()}>
                                                <Send />
                                            </IconButton>
                                        )
                                    }}
                                />
                            </form>
                        </Box>
                    </Paper>
                </Box>

                {/* Sidebar */}
                <Box sx={{
                    flex: 2,
                    overflowY: 'auto', // Adaugă scroll dacă sunt prea multe rezultate
                    maxHeight: 'calc(100vh - 120px)', // Înălțime maximă
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2
                }}>
                    <Paper sx={{ p: 2, flexShrink: 0 }}>
                        <Typography variant="h6" gutterBottom>
                            <Search sx={{ mr: 1, verticalAlign: 'middle' }} />
                            Rezultate Căutare
                        </Typography>
                    </Paper>

                    {searchResults.length > 0 ? (
                        searchResults.map((book, index) => (
                            <BookCover
                                key={index}
                                title={book.title}
                                author={book.authors}
                                summary={book.subjects} // Poți folosi subjects ca un mini-summary
                            />
                        ))
                    ) : (
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="body2" color="text.secondary">
                                Fără rezultate încă...
                            </Typography>
                        </Paper>
                    )}

                    <Paper sx={{ p: 2, flexShrink: 0, mt: 'auto' }}>
                        {/* Biblioteca Noastră rămâne la fel */}
                    </Paper>
                </Box>
            </Container>
        </Box>

    );
}

export default SmartLibrarian;