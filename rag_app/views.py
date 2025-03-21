import os
import tempfile
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from .models import Document, Query
from .forms import DocumentForm, QueryForm
from .rag_utils import (
    initialize_llm,
    initialize_embeddings,
    initialize_vector_store,
    load_from_pdf,
    split_documents,
    create_rag_pipeline
)

# Dictionary to store vector stores by document ID
vector_stores = {}


def index(request):
    documents = Document.objects.all().order_by('-uploaded_at')
    return render(request, 'index.html', {'documents': documents})


def upload_document(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save()

            # Process document and create vector store
            try:
                # Initialize components
                embeddings = initialize_embeddings()
                vector_store = initialize_vector_store(embeddings)

                # Save file to temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    for chunk in document.file.chunks():
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name

                # Load and process PDF
                docs = load_from_pdf(temp_file_path)
                all_splits = split_documents(docs)

                # Add documents to vector store
                vector_store.add_documents(documents=all_splits)

                # Store vector store for later use
                vector_stores[document.id] = vector_store

                # Clean up temporary file
                os.unlink(temp_file_path)

                messages.success(request, 'Document uploaded and processed successfully.')
            except Exception as e:
                document.delete()
                messages.error(request, f'Error processing document: {str(e)}')

            return redirect('index')
    else:
        form = DocumentForm()

    return render(request, 'upload.html', {'form': form})


def query_document(request):
    if request.method == 'POST':
        document_id = request.POST.get('document_id')
        question = request.POST.get('question')

        try:
            document = Document.objects.get(id=document_id)

            # Get vector store or process document if not already processed
            if document_id not in vector_stores:
                # Initialize components
                embeddings = initialize_embeddings()
                vector_store = initialize_vector_store(embeddings)

                # Save file to temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    for chunk in document.file.chunks():
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name

                # Load and process PDF
                docs = load_from_pdf(temp_file_path)
                all_splits = split_documents(docs)

                # Add documents to vector store
                vector_store.add_documents(documents=all_splits)

                # Store vector store
                vector_stores[document_id] = vector_store

                # Clean up temporary file
                os.unlink(temp_file_path)
            else:
                vector_store = vector_stores[document_id]

            # Initialize LLM
            llm = initialize_llm()

            # Create RAG pipeline
            graph = create_rag_pipeline(vector_store, llm)

            # Process query
            response = graph.invoke({"question": question})
            answer = response["answer"]

            # Save query and answer
            query = Query(document=document, question=question, answer=answer)
            query.save()

            return render(request, 'result.html', {
                'document': document,
                'question': question,
                'answer': answer
            })

        except Exception as e:
            messages.error(request, f'Error processing query: {str(e)}')
            return redirect('index')

    return redirect('index')
